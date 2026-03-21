# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CENIC iteration spike diagnostic.

Investigates non-monotonic wall time at N=512 in the wall-vs-N curve by
measuring the inner iteration count K per step_dt call alongside wall time.

Usage:
    uv run python scripts/testing/contact/diag_iteration_spike.py
    uv run python scripts/testing/contact/diag_iteration_spike.py --out-dir /tmp/diag_spike
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model, make_solver

DEFAULT_NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WARMUP_STEPS = 5
TIMED_STEPS = 30
STYLE = {"linewidth": 1.8, "marker": "o", "markersize": 4}


class _LaunchCounter:
    """Wraps wp.capture_launch to count calls between resets."""

    def __init__(self):
        self._real = wp.capture_launch
        self.count = 0

    def __call__(self, graph):
        self.count += 1
        self._real(graph)

    def reset(self):
        self.count = 0

    def install(self):
        wp.capture_launch = self

    def uninstall(self):
        wp.capture_launch = self._real


def _compile_kernels():
    """JIT warmup with N=1 so compilation does not pollute timing."""
    model = build_model(1)
    solver = make_solver(model)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    for _ in range(5):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()


def measure_n(n: int, warmup: int = WARMUP_STEPS, steps: int = TIMED_STEPS) -> dict:
    """Measure iteration count and wall time per step_dt for N worlds.

    Args:
        n: Number of parallel worlds.
        warmup: Warm-up steps (not timed).
        steps: Timed steps.

    Returns:
        Dict with n, naconmax, mean_k, std_k, wall_ms_per_step,
        wall_ms_per_iter, wall_ms_per_sim_s.
    """
    model = build_model(n)
    solver = make_solver(model)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()

    for _ in range(warmup):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()

    counter = _LaunchCounter()
    counter.install()
    k_values = []
    wall_values = []
    try:
        for _ in range(steps):
            counter.reset()
            wp.synchronize()
            t0 = time.perf_counter()
            s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
            wp.synchronize()
            t1 = time.perf_counter()
            k_values.append(counter.count)
            wall_values.append(t1 - t0)
    finally:
        counter.uninstall()

    k_arr = np.array(k_values, dtype=np.float64)
    w_arr = np.array(wall_values, dtype=np.float64)
    mean_k = float(k_arr.mean())
    mean_wall = float(w_arr.mean())

    return {
        "n": n,
        "naconmax": solver.mjw_data.naconmax,
        "mean_k": mean_k,
        "std_k": float(k_arr.std()),
        "wall_ms_per_step": mean_wall * 1e3,
        "wall_ms_per_iter": mean_wall / mean_k * 1e3 if mean_k > 0 else 0.0,
        "wall_ms_per_sim_s": mean_wall / DT_OUTER * 1e3,
    }


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CENIC iteration spike diagnostic")
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=DEFAULT_NS,
        help="World counts to test",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--steps", type=int, default=TIMED_STEPS)
    parser.add_argument("--out-dir", default=".", help="Directory for output plot")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Compiling kernels ...", flush=True)
    _compile_kernels()

    results = []
    for n in args.ns:
        print(f"  N={n:>5} ...", end="", flush=True)
        try:
            r = measure_n(n, warmup=args.warmup, steps=args.steps)
            results.append(r)
            print(
                f"  K={r['mean_k']:.2f}  wall={r['wall_ms_per_step']:.2f} ms/step",
                flush=True,
            )
        except Exception as e:
            print(f"  SKIPPED ({e})", flush=True)

    if not results:
        print("No results collected.", flush=True)
        return

    # --- Table ---
    hdr = f"{'N':>6}  {'naconmax':>8}  {'mean_K':>6}  {'std_K':>5}  {'ms/step':>8}  {'ms/iter':>8}  {'ms/sim_s':>9}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['n']:>6}  {r['naconmax']:>8}  {r['mean_k']:>6.2f}  {r['std_k']:>5.2f}"
            f"  {r['wall_ms_per_step']:>8.2f}  {r['wall_ms_per_iter']:>8.2f}"
            f"  {r['wall_ms_per_sim_s']:>9.1f}"
        )

    # --- 2x2 plot ---
    ns = [r["n"] for r in results]
    mean_ks = [r["mean_k"] for r in results]
    std_ks = [r["std_k"] for r in results]
    wall_step = [r["wall_ms_per_step"] for r in results]
    wall_iter = [r["wall_ms_per_iter"] for r in results]
    naconmaxs = [r["naconmax"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.errorbar(ns, mean_ks, yerr=std_ks, **STYLE, color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Mean K (iterations per step_dt)")
    ax.set_title("Iteration count vs N")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ns, wall_step, **STYLE, color="tab:orange")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time [ms/step_dt]")
    ax.set_title("Wall time per step_dt vs N")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ns, wall_iter, **STYLE, color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time [ms/iteration]")
    ax.set_title("Wall time per iteration vs N")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ns, naconmaxs, **STYLE, color="tab:red")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("naconmax")
    ax.set_title("naconmax vs N (sanity check)")
    ax.grid(True, alpha=0.3)

    _save(fig, out / "diag_iteration_spike.png")


if __name__ == "__main__":
    main()
