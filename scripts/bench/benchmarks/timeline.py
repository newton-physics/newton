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
from scripts.scenes.contact_objects import DT_OUTER, build_model_perturbed, make_solver


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
    model = build_model_perturbed(n_worlds, epsilon=1e-3)
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

    fig, ax = plt.subplots(figsize=(14, 2 + n_worlds * 1.2))

    # Y positions for each world (top to bottom).
    y_positions = list(range(n_worlds - 1, -1, -1))

    # Actual outer step boundaries from recorded data.
    outer_boundaries = [rec["t_start"] for rec in records]
    if records:
        outer_boundaries.append(records[-1]["t_end"])

    # Draw outer step boundaries (tall ticks).
    for t_boundary in outer_boundaries:
        ax.axvline(t_boundary, color="black", linewidth=2.0, zorder=5)

    for rec in records:
        # Group accepted substeps by world (skip rejected).
        by_world: dict[int, list] = {w: [] for w in range(n_worlds)}
        for s in rec["substeps"]:
            if s["accepted"]:
                by_world[s["world"]].append(s)

        for w_idx in range(n_worlds):
            y = y_positions[w_idx]

            for s in by_world[w_idx]:
                # Small tick at each accepted substep boundary.
                ax.plot(
                    [s["sim_time"], s["sim_time"]], [y - 0.3, y + 0.3],
                    color="black", linewidth=0.8, zorder=4,
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"World {w}" for w in range(n_worlds)], fontsize=10)
    ax.set_xlabel("Simulation time [s]", fontsize=11)
    ax.set_title(
        f"Adaptive substep timeline  ({n_worlds} worlds, DT_outer={dt_outer * 1e3:.0f} ms, tol=1e-3)",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)

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
