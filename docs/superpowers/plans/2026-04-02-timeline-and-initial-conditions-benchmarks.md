# Timeline & Initial Conditions Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new benchmark modules: (1) a timeline graphic showing per-world inner substeps across outer step boundaries, and (2) a comparison of N-scaling behavior under identical vs. deterministically perturbed initial conditions.

**Architecture:** Both benchmarks follow the existing pattern in `scripts/bench/benchmarks/` -- each is a standalone module with `run()`, `plot()`, and `main()` functions, registered in the bench runner. The timeline module instruments the solver's inner loop with `.numpy()` readbacks for diagnostic data collection (never called in performance paths). The initial conditions module reuses the scaling benchmark's measurement infrastructure with a new `build_model_perturbed()` scene factory that applies deterministic per-world offsets to `joint_q`.

**Tech Stack:** Python, matplotlib, warp, numpy, existing `scripts.bench.infra` and `scripts.bench.plotting` infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/bench/benchmarks/timeline.py` | Create | Timeline diagnostic: collect per-world per-iteration dt/accepted data, produce timeline plot |
| `scripts/bench/benchmarks/initial_conditions.py` | Create | Run scaling benchmark under identical vs. perturbed ICs, produce 5 comparison plots |
| `scripts/scenes/contact_objects.py` | Modify | Add `build_model_perturbed(n_worlds)` that applies deterministic z-offsets to each world's bodies |
| `scripts/bench/plotting.py` | Modify | Add STYLES entries for `"identical"` and `"perturbed"` series |
| `scripts/bench/runner.py` | Modify | Register `"timeline"` and `"initial_conditions"` in `BENCHMARK_NAMES` |

---

### Task 1: Add `build_model_perturbed` to contact_objects scene

**Files:**
- Modify: `scripts/scenes/contact_objects.py`

- [ ] **Step 1: Add `build_model_perturbed` function**

Add after the existing `build_model` function (line ~99):

```python
def build_model_perturbed(n_worlds: int, epsilon: float = 1e-4) -> newton.Model:
    """N replicated worlds with deterministic per-world z-perturbation.

    Each world's bodies get z-offset = world_index * epsilon [m].
    World 0 is unperturbed (identical to build_model output).
    """
    model = build_model(n_worlds)

    # Perturb joint_q on CPU, then write back.
    joint_q_np = model.joint_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            z_idx = w * coords_per_world + b * 7 + 2  # z-component of position
            joint_q_np[z_idx] += offset

    model.joint_q.assign(joint_q_np)

    # Also perturb body_q (used by renderer / solver sync).
    body_q_np = model.body_q.numpy()
    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            body_idx = w * bodies_per_world + b
            body_q_np[body_idx][2] += offset  # z component of position in transform

    model.body_q.assign(body_q_np)

    return model
```

- [ ] **Step 2: Verify it runs**

Run:
```bash
uv run python -c "
from scripts.scenes.contact_objects import build_model, build_model_perturbed
m1 = build_model(4)
m2 = build_model_perturbed(4)
import numpy as np
q1 = m1.joint_q.numpy()
q2 = m2.joint_q.numpy()
cpw = m1.joint_coord_count // 4
for w in range(4):
    z0 = q1[w * cpw + 2]
    z1 = q2[w * cpw + 2]
    print(f'world {w}: z_original={z0:.6f}  z_perturbed={z1:.6f}  diff={z1-z0:.6e}')
"
```

Expected: world 0 has diff=0, world 1 has diff=1e-4, world 2 has diff=2e-4, world 3 has diff=3e-4.

---

### Task 2: Add STYLES entries for initial conditions series

**Files:**
- Modify: `scripts/bench/plotting.py`

- [ ] **Step 1: Add style entries**

Add to the `STYLES` dict (after the existing `"single_iter"` entry, around line 39):

```python
    "identical": PlotStyle("#2ca02c", "^", "-", "Identical ICs"),
    "perturbed": PlotStyle("#9467bd", "v", "-", "Perturbed ICs"),
```

---

### Task 3: Create timeline benchmark module

**Files:**
- Create: `scripts/bench/benchmarks/timeline.py`

- [ ] **Step 1: Create the timeline module**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Timeline benchmark: per-world inner substep visualization.

Diagnostic-only -- uses .numpy() readbacks between iterations to log
per-world dt, sim_time, and accept/reject at each inner substep.
Not used for performance measurement.

Standalone:
    uv run python -m scripts.bench.benchmarks.timeline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warp as wp

from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver


@dataclass
class SubstepRecord:
    """One inner iteration's snapshot for one world."""
    world: int
    iteration: int  # iteration index within this outer step
    dt: float       # timestep used for this attempt
    sim_time: float # sim_time after this iteration
    accepted: bool  # whether the step was accepted


@dataclass
class OuterStepRecord:
    """All inner iterations for one outer step."""
    outer_step: int
    t_start: float  # sim_time at start of this outer step
    t_end: float    # t_start + DT_OUTER
    substeps: list[SubstepRecord] = field(default_factory=list)


def _collect_timeline(
    n_worlds: int,
    warmup: int,
    outer_steps: int,
) -> dict:
    """Run solver with per-iteration readbacks to build timeline data.

    This is a diagnostic function -- .numpy() is called after every inner
    iteration to log per-world state. Never use for performance measurement.
    """
    model = build_model(n_worlds)
    solver = make_solver(model)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()

    # Warmup into contact regime (no logging).
    for _ in range(warmup):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)

    # Now collect detailed timeline data by manually driving the inner loop.
    # This mirrors step_dt logic but with .numpy() readbacks per iteration.
    records: list[OuterStepRecord] = []

    for step_idx in range(outer_steps):
        device = solver.model.device
        n = solver.model.world_count
        effective_dt_max = min(solver._dt_max, DT_OUTER)

        # --- Begin step_dt preamble ---
        from newton._src.solvers.mujoco.solver_mujoco_cenic import (
            _apply_dt_cap,
            _boundary_advance,
        )

        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[solver._ideal_dt, solver._dt_min, effective_dt_max,
                    solver._dt, solver._dt_half],
            device=device,
        )

        wp.copy(solver._state_cur.joint_q, s0.joint_q)
        wp.copy(solver._state_cur.joint_qd, s0.joint_qd)
        if s0.body_q is not None and solver._state_cur.body_q is not None:
            wp.copy(solver._state_cur.body_q, s0.body_q)
        if s0.body_qd is not None and solver._state_cur.body_qd is not None:
            wp.copy(solver._state_cur.body_qd, s0.body_qd)

        solver._apply_mjc_control(solver.model, s0, ctrl, solver.mjw_data)
        solver._enable_rne_postconstraint(solver._state_cur)

        wp.launch(_boundary_advance, dim=n,
                  inputs=[solver._next_time, DT_OUTER], device=device)

        solver._iteration_count_buf.fill_(0)
        solver._boundary_flag.fill_(1)

        t_start = float(solver._sim_time.numpy()[0])
        rec = OuterStepRecord(
            outer_step=step_idx,
            t_start=t_start,
            t_end=t_start + DT_OUTER,
        )

        # --- Inner loop with readbacks ---
        iteration = 0
        while True:
            solver._run_iteration_body(effective_dt_max)

            # Diagnostic readbacks (PCIe transfers -- diagnostic only).
            dt_np = solver._dt.numpy()
            sim_time_np = solver._sim_time.numpy()
            accepted_np = solver._accepted.numpy()
            boundary_done = solver._boundary_flag.numpy()[0] == 0

            for w in range(n_worlds):
                rec.substeps.append(SubstepRecord(
                    world=w,
                    iteration=iteration,
                    dt=float(dt_np[w]),
                    sim_time=float(sim_time_np[w]),
                    accepted=bool(accepted_np[w]),
                ))

            iteration += 1
            if boundary_done:
                break

        # --- Finalize (copy state back) ---
        wp.copy(s0.joint_q, solver._state_cur.joint_q)
        wp.copy(s0.joint_qd, solver._state_cur.joint_qd)
        if s0.body_q is not None and solver._state_cur.body_q is not None:
            wp.copy(s0.body_q, solver._state_cur.body_q)
        if s0.body_qd is not None and solver._state_cur.body_qd is not None:
            wp.copy(s0.body_qd, solver._state_cur.body_qd)

        records.append(rec)

    # Serialize to JSON-friendly format.
    data = {
        "n_worlds": n_worlds,
        "warmup": warmup,
        "outer_steps": outer_steps,
        "dt_outer": DT_OUTER,
        "records": [],
    }
    for rec in records:
        entry = {
            "outer_step": rec.outer_step,
            "t_start": rec.t_start,
            "t_end": rec.t_end,
            "substeps": [
                {
                    "world": s.world,
                    "iteration": s.iteration,
                    "dt": s.dt,
                    "sim_time": s.sim_time,
                    "accepted": s.accepted,
                }
                for s in rec.substeps
            ],
        }
        data["records"].append(entry)

    return data


def run(
    n_worlds: int = 4,
    warmup: int = 50,
    outer_steps: int = 5,
) -> dict:
    """Collect timeline data."""
    print(f"  Timeline: {n_worlds} worlds, {warmup} warmup, {outer_steps} outer steps", flush=True)
    return _collect_timeline(n_worlds, warmup, outer_steps)


def plot(data: dict, out_dir: Path) -> None:
    """Generate timeline plot: one row per world, inner substeps as segments."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_worlds = data["n_worlds"]
    records = data["records"]
    dt_outer = data["dt_outer"]

    # Color palette for worlds.
    world_colors = plt.cm.tab10.colors[:n_worlds]

    fig, ax = plt.subplots(figsize=(14, 2 + n_worlds * 1.2))

    # Y positions for each world (top to bottom).
    y_positions = list(range(n_worlds - 1, -1, -1))

    for rec in records:
        t_start = rec["t_start"]
        t_end = rec["t_end"]

        # Draw outer step boundary (large tick).
        for w_idx in range(n_worlds):
            y = y_positions[w_idx]
            ax.plot(
                [t_start, t_start], [y - 0.35, y + 0.35],
                color="black", linewidth=2.0, zorder=5,
            )

        # Group substeps by world.
        by_world: dict[int, list] = {w: [] for w in range(n_worlds)}
        for s in rec["substeps"]:
            by_world[s["world"]].append(s)

        for w_idx in range(n_worlds):
            y = y_positions[w_idx]
            color = world_colors[w_idx]
            world_substeps = by_world[w_idx]

            for s in world_substeps:
                # Each accepted substep is a horizontal segment ending at sim_time.
                # Width = dt used for this step.
                seg_end = s["sim_time"]
                seg_start = seg_end - s["dt"] if s["accepted"] else seg_end

                if s["accepted"]:
                    # Accepted: solid colored segment.
                    ax.barh(
                        y, s["dt"], left=seg_start, height=0.5,
                        color=color, alpha=0.7, edgecolor="black",
                        linewidth=0.5, zorder=3,
                    )
                    # Small tick at substep boundary.
                    ax.plot(
                        [seg_end, seg_end], [y - 0.2, y + 0.2],
                        color="black", linewidth=0.8, zorder=4,
                    )
                else:
                    # Rejected: red hatched marker at the dt that was tried.
                    ax.barh(
                        y, s["dt"], left=seg_end - s["dt"], height=0.5,
                        color="red", alpha=0.3, edgecolor="red",
                        linewidth=0.5, linestyle="--", zorder=2,
                    )

    # Draw final outer boundary.
    if records:
        last_t_end = records[-1]["t_end"]
        for w_idx in range(n_worlds):
            y = y_positions[w_idx]
            ax.plot(
                [last_t_end, last_t_end], [y - 0.35, y + 0.35],
                color="black", linewidth=2.0, zorder=5,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"World {w}" for w in range(n_worlds)], fontsize=10)
    ax.set_xlabel("Simulation time [s]", fontsize=11)
    ax.set_title(
        f"Adaptive substep timeline  ({n_worlds} worlds, DT_outer={dt_outer * 1e3:.0f} ms, tol=1e-3)",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Legend.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=world_colors[0], alpha=0.7, edgecolor="black", label="Accepted substep"),
        Patch(facecolor="red", alpha=0.3, edgecolor="red", label="Rejected substep"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    save_fig(fig, out_dir / "timeline_substeps.png")


def main():
    parser = argparse.ArgumentParser(description="Timeline benchmark")
    parser.add_argument("--n-worlds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--outer-steps", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(args.n_worlds, args.warmup, args.outer_steps)

    with open(out_dir / "timeline.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'timeline.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
```

Note: The step_dt preamble is replicated here to allow per-iteration readbacks. The `_run_iteration_body()` call is identical to what `step_dt` uses -- the only difference is that we read back `.numpy()` between iterations instead of only checking the boundary flag.

- [ ] **Step 2: Verify it runs**

Run:
```bash
uv run python -m scripts.bench.benchmarks.timeline --warmup 50 --outer-steps 3 --n-worlds 2
```

Expected: JSON output with substep records, plot saved to `scripts/bench/results/plots/timeline_substeps.png`.

---

### Task 4: Create initial conditions benchmark module

**Files:**
- Create: `scripts/bench/benchmarks/initial_conditions.py`

- [ ] **Step 1: Create the initial conditions module**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial conditions benchmark: identical vs. perturbed IC scaling comparison.

Produces the same 5 plots as the scaling benchmark but comparing two
initial condition regimes instead of stepping modes.

Standalone:
    uv run python -m scripts.bench.benchmarks.initial_conditions
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
    make_solver,
)

IC_MODES = ["identical", "perturbed"]


def _measure_ic_mode(
    mode: str, n: int, steps: int, warmup: int, trials: int = 1,
) -> MeasureResult:
    """Measure CENIC at one N with one IC mode."""
    build_fn = build_model if mode == "identical" else build_model_perturbed

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


def _collect_error_trace(mode: str, steps: int) -> dict:
    """Run N=1 and record accepted error + dt at each outer step."""
    build_fn = build_model if mode == "identical" else build_model_perturbed
    model = build_fn(1)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    sim_times, errors, dts = [], [], []
    for _ in range(steps):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        sim_times.append(float(solver.sim_time.numpy()[0]))
        errors.append(float(solver.last_error.numpy()[0]))
        dts.append(float(solver.dt.numpy()[0]))

    return {"sim_times": sim_times, "errors": errors, "dts": dts, "tol": solver._tol}


def run(ns: list[int], steps: int, warmup: int, trials: int = 1) -> dict:
    """Run both IC modes at all N values."""
    data: dict = {
        "ns": ns, "steps": steps, "warmup": warmup,
        "trials": trials, "modes": {},
    }

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

    # Exponents: wall time for both modes (CENIC adaptive).
    data["exponents"] = {}
    for mode in IC_MODES:
        data["exponents"][mode] = power_law_exponent(
            ns, data["modes"][mode]["medians"],
        )

    # Error traces (N=1, both modes).
    print("  Collecting error traces (N=1)...", flush=True)
    data["error_traces"] = {}
    for mode in IC_MODES:
        data["error_traces"][mode] = _collect_error_trace(mode, steps + warmup)

    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate 5 comparison plots from results dict."""
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


def main():
    parser = argparse.ArgumentParser(description="Initial conditions benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup, args.trials)

    with open(out_dir / "initial_conditions.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'initial_conditions.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs with small N**

Run:
```bash
uv run python -m scripts.bench.benchmarks.initial_conditions --ns 1 4 --steps 10 --warmup 10
```

Expected: prints summary table, saves JSON + 5 plots.

---

### Task 5: Register benchmarks in the runner

**Files:**
- Modify: `scripts/bench/runner.py`

- [ ] **Step 1: Add to BENCHMARK_NAMES**

Change line 27:
```python
BENCHMARK_NAMES = ["scaling", "components", "accuracy"]
```
to:
```python
BENCHMARK_NAMES = ["scaling", "components", "accuracy", "timeline", "initial_conditions"]
```

- [ ] **Step 2: Handle CLI arg forwarding for new benchmarks**

In `_run_benchmark_subprocess`, the existing arg-forwarding logic gates `ns`, `steps`, `warmup` on `bench_name in ("scaling", "components")`. Update line 74-80 to also include the new benchmarks:

Change:
```python
    if "ns" in args and bench_name in ("scaling", "components"):
```
to:
```python
    if "ns" in args and bench_name in ("scaling", "components", "initial_conditions"):
```

Change:
```python
    if "steps" in args and bench_name in ("scaling", "components"):
        cmd.extend(["--steps", str(args["steps"])])
    if "warmup" in args and bench_name in ("scaling", "components"):
        cmd.extend(["--warmup", str(args["warmup"])])
```
to:
```python
    if "steps" in args and bench_name in ("scaling", "components", "initial_conditions"):
        cmd.extend(["--steps", str(args["steps"])])
    if "warmup" in args and bench_name in ("scaling", "components", "initial_conditions"):
        cmd.extend(["--warmup", str(args["warmup"])])
```

- [ ] **Step 3: Verify registration**

Run:
```bash
uv run -m scripts.bench --list
```

Expected: shows all 5 benchmarks including `timeline` and `initial_conditions`.

---

### Task 6: End-to-end verification

- [ ] **Step 1: Run timeline benchmark standalone**

```bash
uv run python -m scripts.bench.benchmarks.timeline --warmup 50 --outer-steps 5 --n-worlds 4
```

Verify: `scripts/bench/results/plots/timeline_substeps.png` exists and shows 4 world rows with colored substep segments.

- [ ] **Step 2: Run initial conditions benchmark standalone**

```bash
uv run python -m scripts.bench.benchmarks.initial_conditions --ns 1 4 16 --steps 20 --warmup 20
```

Verify: 5 plots saved under `scripts/bench/results/plots/ic_*.png`.

- [ ] **Step 3: Run both via the bench runner**

```bash
uv run -m scripts.bench --only timeline
uv run -m scripts.bench --only initial_conditions --ns 1 4 16 --steps 20 --warmup 20
```

Verify: results saved under `scripts/bench/results/<git-hash>/plots/`.
