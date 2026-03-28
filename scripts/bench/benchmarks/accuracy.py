# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmark: wall time vs tolerance, error traces, dt traces.

Each measurement runs in a subprocess for GPU state isolation.

Standalone:
    uv run python -m scripts.bench.benchmarks.accuracy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_INNER_MIN, DT_OUTER


def _run_in_subprocess(code: str) -> str:
    """Run Python code in a fresh subprocess, return last stdout line."""
    env = {**__import__("os").environ, "WARP_LOG_LEVEL": "error"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
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
        from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver
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


def _measure_traces(tol: float, n_worlds: int, sim_duration: float) -> dict:
    """Returns {t, e, d} for world 0."""
    code = textwrap.dedent(f"""\
        import json, warp as wp
        from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver
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


def run(
    tols: list[float] | None = None,
    sim_duration: float = 2.0,
    trials: int = 3,
) -> dict:
    """Run accuracy benchmarks."""
    if tols is None:
        tols = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    data: dict = {
        "tols": tols,
        "sim_duration": sim_duration,
        "dt_outer": DT_OUTER,
        "dt_inner_min": DT_INNER_MIN,
    }

    # Wall time vs tol
    print(f"\n[1/2] Wall time vs tol  N=1  sim={sim_duration}s  trials={trials}", flush=True)
    wall_vs_tol = []
    for tol in tols:
        ms = _measure_cenic(1, tol, sim_duration, trials)
        print(f"  tol={tol:.0e}  {ms:.1f} ms/sim-s", flush=True)
        wall_vs_tol.append(ms)
    data["wall_vs_tol"] = wall_vs_tol

    # Error and dt traces
    print("\n[2/2] Error & dt traces  tol=1e-3  N=1", flush=True)
    traces = _measure_traces(1e-3, 1, sim_duration)
    data["traces"] = traces
    print(f"  {len(traces['t'])} outer steps", flush=True)

    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate 3 accuracy plots."""
    tols = data["tols"]
    dt_inner_min = data["dt_inner_min"]
    out_dir.mkdir(parents=True, exist_ok=True)

    style = {"linewidth": 1.8, "marker": "o", "markersize": 4}

    # Fig 1: Wall time vs tol
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tols, data["wall_vs_tol"], **style, color="tab:blue")
    ax.set_xlabel("Error tolerance")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    ax.set_title("Wall time vs tolerance  (N=1, 2 s sim including contact)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "accuracy_wall_vs_tol.png")

    # Fig 2: Error trace
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
    save_fig(fig, out_dir / "accuracy_error_trace.png")

    # Fig 3: dt trace
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, dts_trace * 1e3, color="tab:orange", linewidth=1.2)
    ax.axhline(
        dt_inner_min * 1e3, color="grey", linestyle=":", linewidth=1.0,
        label=f"dt_inner_min = {dt_inner_min * 1e3:.2f} ms",
    )
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("dt_inner [ms]")
    ax.set_title(f"Adaptive inner step size over time  (N=1, tol={trace_tol:.0e})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "accuracy_dt_trace.png")


def main():
    parser = argparse.ArgumentParser(description="Accuracy benchmark")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sim-duration", type=float, default=2.0)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data = run(trials=args.trials, sim_duration=args.sim_duration)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'accuracy.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
