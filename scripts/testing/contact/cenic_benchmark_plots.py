"""CENIC benchmark plots.

Generates 6 figures: wall time vs tol, wall time vs N, GPU amortization,
inf-norm q error trace, adaptive dt trace, and GPU component breakdown.

Each benchmark point runs in a subprocess to prevent GPU state leakage.
Results are saved to JSON so benchmarks and plotting can run separately.

Usage:
    # Run everything (CENIC only):
    uv run python scripts/testing/contact/cenic_benchmark_plots.py

    # Run CENIC + fixed baselines:
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --compare-fixed

    # Benchmark only (save JSON, no plots):
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --bench-only

    # Plot only (from saved JSON):
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --plot-only

    # Faster iteration (1 trial instead of 3):
    uv run python scripts/testing/contact/cenic_benchmark_plots.py --trials 1
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.testing.contact.cenic_contact_objects import DT_OUTER, DT_INNER_MIN


def _run_in_subprocess(code: str) -> str:
    """Run Python code in a fresh subprocess, return last stdout line."""
    env = {**__import__("os").environ, "WARP_LOG_LEVEL": "error"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600, env=env,
    )
    if result.returncode != 0:
        print(f"  SUBPROCESS STDERR:\n{result.stderr[-500:]}", flush=True)
        raise RuntimeError(f"Subprocess failed (exit {result.returncode})")
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    return lines[-1]


def _measure_cenic(n: int, tol: float, sim_duration: float, trials: int) -> float:
    """Measure CENIC wall time (ms/sim-s), median of `trials` fresh processes."""
    code = textwrap.dedent(f"""\
        import time, warp as wp
        from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model, make_solver
        model = build_model({n})
        solver = make_solver(model, tol={tol})
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        for _ in range(5):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        n_steps = round({sim_duration} / DT_OUTER)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        print(elapsed / {sim_duration} * 1e3)
    """)
    samples = [float(_run_in_subprocess(code)) for _ in range(trials)]
    return float(np.median(samples))


def _measure_fixed(n: int, dt: float, sim_duration: float, trials: int) -> float:
    """Measure fixed-step wall time (ms/sim-s), median of `trials` fresh processes."""
    code = textwrap.dedent(f"""\
        import time, warp as wp
        import newton.solvers
        from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model
        model = build_model({n})
        solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True, nconmax=128, njmax=640)
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        contacts = model.contacts()
        n_outer = round({sim_duration} / DT_OUTER)
        n_inner = round(DT_OUTER / {dt})
        for _ in range(3):
            for _ in range(n_inner):
                s1 = solver.step(s0, s1, ctrl, contacts, {dt})
                s0, s1 = s1, s0
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_outer):
            for _ in range(n_inner):
                s1 = solver.step(s0, s1, ctrl, contacts, {dt})
                s0, s1 = s1, s0
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        print(elapsed / {sim_duration} * 1e3)
    """)
    samples = [float(_run_in_subprocess(code)) for _ in range(trials)]
    return float(np.median(samples))


def _measure_tol(tol: float, n_worlds: int, sim_duration: float, trials: int) -> float:
    """Measure CENIC wall time (ms/sim-s) for one tol, median of `trials` fresh processes."""
    code = textwrap.dedent(f"""\
        import time, warp as wp
        from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model, make_solver
        model = build_model({n_worlds})
        solver = make_solver(model, tol={tol})
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        for _ in range(5):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        n_steps = round({sim_duration} / DT_OUTER)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        print(elapsed / {sim_duration} * 1e3)
    """)
    samples = [float(_run_in_subprocess(code)) for _ in range(trials)]
    return float(np.median(samples))


def _measure_traces(tol: float, n_worlds: int, sim_duration: float):
    """Returns (sim_times, errors, dts) for world 0."""
    code = textwrap.dedent(f"""\
        import json, warp as wp
        from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model, make_solver
        model = build_model({n_worlds})
        solver = make_solver(model, tol={tol})
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        sim_times, errors, dts = [], [], []
        for _ in range(round({sim_duration} / DT_OUTER)):
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
            sim_times.append(float(solver.sim_time.numpy()[0]))
            errors.append(float(solver.last_error.numpy()[0]))
            dts.append(float(solver.dt.numpy()[0]))
        print(json.dumps({{"t": sim_times, "e": errors, "d": dts}}))
    """)
    return json.loads(_run_in_subprocess(code))


def _measure_diag(n: int, steps: int):
    """Measure GPU component breakdown for one N in a fresh process."""
    code = textwrap.dedent(f"""\
        import json
        from scripts.testing.contact.cenic_scaling_diag import measure_n
        r = measure_n({n}, steps={steps}, warmup=20)
        print(json.dumps(r))
    """)
    return json.loads(_run_in_subprocess(code))


# ── Benchmarking ──────────────────────────────────────────────────────

def run_benchmarks(ns, tols, sim_duration, trials, compare_fixed, outer_steps):
    """Run all benchmarks, return results dict."""
    data = {
        "ns": ns, "tols": tols, "sim_duration": sim_duration,
        "dt_outer": DT_OUTER, "dt_inner_min": DT_INNER_MIN,
    }

    print(f"\n[Fig 1] Wall time vs tol  N=1  sim={sim_duration}s  trials={trials}", flush=True)
    wall_vs_tol = []
    for tol in tols:
        ms = _measure_tol(tol, 1, sim_duration, trials)
        print(f"  tol={tol:.0e}  {ms:.1f} ms/sim-s", flush=True)
        wall_vs_tol.append(ms)
    data["wall_vs_tol"] = wall_vs_tol

    print(f"\n[Fig 2] Wall time vs N  tol=1e-3  sim={sim_duration}s  trials={trials}", flush=True)
    wall_vs_n = []
    for n in ns:
        ms = _measure_cenic(n, 1e-3, sim_duration, trials)
        print(f"  N={n:>5}  {ms:.1f} ms/sim-s", flush=True)
        wall_vs_n.append(ms)
    data["wall_vs_n"] = wall_vs_n

    if compare_fixed:
        dt_coarse = DT_OUTER
        dt_fine = DT_OUTER / 20
        data["dt_coarse"] = dt_coarse
        data["dt_fine"] = dt_fine

        print(f"\n[Fig 2 fixed coarse] dt={dt_coarse:.0e}  trials={trials}", flush=True)
        wall_fixed_coarse = []
        for n in ns:
            ms = _measure_fixed(n, dt_coarse, sim_duration, trials)
            print(f"  N={n:>5}  {ms:.1f} ms/sim-s", flush=True)
            wall_fixed_coarse.append(ms)
        data["wall_fixed_coarse"] = wall_fixed_coarse

        print(f"\n[Fig 2 fixed fine] dt={dt_fine:.0e}  trials={trials}", flush=True)
        wall_fixed_fine = []
        for n in ns:
            ms = _measure_fixed(n, dt_fine, sim_duration, trials)
            print(f"  N={n:>5}  {ms:.1f} ms/sim-s", flush=True)
            wall_fixed_fine.append(ms)
        data["wall_fixed_fine"] = wall_fixed_fine

    print(f"\n[Fig 4/5] Error & dt traces  tol=1e-3  N=1", flush=True)
    traces = _measure_traces(1e-3, 1, sim_duration)
    data["traces"] = traces
    print(f"  {len(traces['t'])} outer steps", flush=True)

    print(f"\n[Fig 6] GPU component breakdown  steps={outer_steps}", flush=True)
    diag_results = []
    for n in ns:
        r = _measure_diag(n, outer_steps)
        print(f"  N={n:>5}  step_dt={r['step_dt']*1e3:.2f} ms", flush=True)
        diag_results.append(r)
    data["diag"] = diag_results

    return data


# ── Plotting ──────────────────────────────────────────────────────────

STYLE = {"linewidth": 1.8, "marker": "o", "markersize": 4}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}", flush=True)


def plot_all(data, out):
    """Generate all figures from a results dict."""
    ns = data["ns"]
    tols = data["tols"]
    dt_outer = data["dt_outer"]
    dt_inner_min = data["dt_inner_min"]

    # Fig 1: wall time vs tol
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tols, data["wall_vs_tol"], **STYLE, color="tab:blue")
    ax.set_xlabel("Error tolerance")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    ax.set_title("Wall time vs tolerance  (N=1, 2 s sim including contact)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig1_wall_vs_tol.png")

    # Fig 2: wall time vs N
    wall_vs_n = data["wall_vs_n"]
    has_fixed = "wall_fixed_coarse" in data

    sat = 128
    mask = [i for i, n in enumerate(ns) if n <= sat]
    ns_fit = np.array([ns[i] for i in mask])
    ws_fit = np.array([wall_vs_n[i] for i in mask])
    slope, intercept = np.polyfit(np.log(ns_fit), np.log(ws_fit), 1)
    fit_x = np.geomspace(ns_fit[0], ns_fit[-1], 100)
    fit_y = np.exp(intercept) * fit_x ** slope

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, wall_vs_n, **STYLE, color="black", label="CENIC adaptive (tol=1e-3)")
    if has_fixed:
        dt_coarse = data["dt_coarse"]
        dt_fine = data["dt_fine"]
        ax.plot(ns, data["wall_fixed_coarse"], **STYLE, color="tab:green",
                label=f"Fixed dt={dt_coarse*1e3:.0f} ms (coarse)")
        ax.plot(ns, data["wall_fixed_fine"], **STYLE, color="tab:red",
                label=f"Fixed dt={dt_fine*1e3:.1f} ms (fine)")
        for wall_fixed, dt_val, color in [
            (data["wall_fixed_coarse"], dt_coarse, "tab:green"),
            (data["wall_fixed_fine"], dt_fine, "tab:red"),
        ]:
            mask_f = [i for i, n in enumerate(ns) if n <= sat]
            ns_f = np.array([ns[i] for i in mask_f])
            ws_f = np.array([wall_fixed[i] for i in mask_f])
            sl_f, ic_f = np.polyfit(np.log(ns_f), np.log(ws_f), 1)
            fit_xf = np.geomspace(ns_f[0], ns_f[-1], 100)
            fit_yf = np.exp(ic_f) * fit_xf ** sl_f
            ax.plot(fit_xf, fit_yf, color=color, linestyle="--", linewidth=1.2,
                    label=f"fixed dt={dt_val*1e3:.1g} ms fit: t ~ N^{{{sl_f:.2f}}}")
    ax.plot(fit_x, fit_y, color="tab:blue", linestyle="--", linewidth=1.2,
            label=f"CENIC fit N <= {sat}: t ~ N^{{{slope:.2f}}}")
    ax.axvline(sat, color="grey", linestyle=":", linewidth=0.8, label=f"saturation N={sat}")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    title = "Fixed vs adaptive wall time" if has_fixed else "Wall time vs N worlds"
    ax.set_title(f"{title}  (tol=1e-3, 2 s sim including contact)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig2_wall_vs_n.png")

    # Fig 3: cost per world
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
               label=f"optimal N = {ns[opt_idx]}")
    ax.legend(fontsize=8)
    _save(fig, out / "fig3_cost_per_world.png")

    # Fig 4: error trace
    traces = data["traces"]
    ts = np.array(traces["t"])
    errs = np.array(traces["e"])
    dts_trace = np.array(traces["d"])
    trace_tol = 1e-3

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, errs, color="tab:blue", linewidth=1.2)
    ax.axhline(trace_tol, color="tab:blue", linestyle="--", linewidth=0.8,
               label=f"tolerance = {trace_tol:.0e}")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Inf-norm q error (world 0)")
    ax.set_title(f"Integration error over time  (N=1, tol={trace_tol:.0e})")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig4_error_over_time.png")

    # Fig 5: dt trace
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, dts_trace * 1e3, color="tab:orange", linewidth=1.2)
    ax.axhline(dt_inner_min * 1e3, color="grey", linestyle=":", linewidth=1.0,
               label=f"dt_inner_min = {dt_inner_min*1e3:.2f} ms")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("dt_inner [ms]")
    ax.set_title(f"Adaptive inner step size over time  (N=1, tol={trace_tol:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig5_dt_over_time.png")

    # Fig 6: GPU component breakdown
    diag_results = data["diag"]
    _COMPONENTS = [
        ("3x_substep",      "3x MuJoCo step (uncaptured)",         "tab:blue",   "-",  2.0),
        ("mujoco_warp",     "  mujoco_warp kernel (x1)",           "tab:cyan",   "--", 1.2),
        ("graph_replay",    "CUDA graph replay (full inner step)",  "tab:orange", "-",  1.5),
        ("update_mjc_data", "update_mjc_data (x1)",                "tab:green",  "--", 1.2),
        ("update_newton",   "update_newton_state (x1)",            "tab:red",    "--", 1.2),
        ("boundary_numpy",  "boundary_flag.numpy() -- 1 int32",    "tab:purple", ":",  1.8),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color, ls, lw in _COMPONENTS:
        ys = [r[key] * 1e3 for r in diag_results]
        ax.plot(ns, ys, label=label, color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=4)
    ax.set_xlabel("N parallel worlds")
    ax.set_ylabel("Wall time per step_dt call [ms]")
    ax.set_title("GPU component breakdown: MuJoCo physics kernel vs device-transfer overhead")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out / "fig6_component_breakdown.png")

    print("\nAll plots saved.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="scripts/testing/contact/bench_results",
                        help="Directory for PNGs and JSON")
    parser.add_argument("--compare-fixed", action="store_true",
                        help="Also benchmark fixed-step baselines")
    parser.add_argument("--trials", type=int, default=3,
                        help="Median-of-N trials per measurement (default 3, use 1 for speed)")
    parser.add_argument("--bench-only", action="store_true",
                        help="Run benchmarks and save JSON, skip plotting")
    parser.add_argument("--plot-only", action="store_true",
                        help="Plot from saved JSON, skip benchmarks")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "bench_data.json"

    ns           = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000]
    tols         = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    sim_duration = 2.0
    outer_steps  = 200

    if args.plot_only:
        print(f"Loading data from {json_path}", flush=True)
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = run_benchmarks(
            ns, tols, sim_duration, args.trials, args.compare_fixed, outer_steps,
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nData saved -> {json_path}", flush=True)

    if not args.bench_only:
        plot_all(data, out)


if __name__ == "__main__":
    main()
