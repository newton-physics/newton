# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel component breakdown benchmark.

Instruments each operation in the CENIC iteration body with
wp.synchronize() barriers to identify which kernels scale with N.

Standalone:
    uv run python -m scripts.bench.benchmarks.components --ns 1 4 16 64 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import warp as wp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.infra import power_law_exponent
from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized, make_solver
from newton._src.solvers.mujoco.solver_mujoco_cenic import (
    _apply_dt_cap,
    _advance_sim_time,
    _boundary_advance,
    _boundary_check,
    _boundary_reset,
    _calc_adjusted_step,
    _inf_norm_state_error_kernel,
    _select_float_kernel,
    _select_spatial_vector_kernel,
    _select_transform_kernel,
)

ITER_COMPONENTS = [
    "snapshot_copies",
    "substep1_update_mjc", "substep1_dt_copy", "substep1_mujoco", "substep1_update_newton",
    "substep2_update_mjc", "substep2_dt_copy", "substep2_mujoco", "substep2_update_newton",
    "substep3_update_mjc", "substep3_dt_copy", "substep3_mujoco", "substep3_update_newton",
    "inf_norm_error", "calc_adjusted_step",
    "select_joint_q", "select_joint_qd", "select_body_q", "select_body_qd",
    "advance_sim_time", "apply_dt_cap", "boundary_check",
]

PRE_POST_COMPONENTS = ["pre_state_copy_in", "apply_mjc_control", "post_state_copy_out"]
ALL_COMPONENTS = ITER_COMPONENTS + PRE_POST_COMPONENTS

PLOT_GROUPS = {
    "mujoco_warp_x3": ["substep1_mujoco", "substep2_mujoco", "substep3_mujoco"],
    "update_mjc_x3": ["substep1_update_mjc", "substep2_update_mjc", "substep3_update_mjc"],
    "dt_copy_x3": ["substep1_dt_copy", "substep2_dt_copy", "substep3_dt_copy"],
    "update_newton_x3": ["substep1_update_newton", "substep2_update_newton", "substep3_update_newton"],
    "error_control": ["inf_norm_error", "calc_adjusted_step"],
    "state_select_x4": ["select_joint_q", "select_joint_qd", "select_body_q", "select_body_qd"],
    "bookkeeping": ["advance_sim_time", "apply_dt_cap", "boundary_check"],
    "pre_post": PRE_POST_COMPONENTS,
}


def _measure_n(n: int, steps: int, warmup: int) -> dict:
    """Time each CENIC sub-component for N worlds.

    Phase 1: step_dt end-to-end + iteration count K.
    Phase 2: per-component manual timing with sync barriers.
    """
    model = build_model_randomized(n)
    solver = make_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    for _ in range(warmup):
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
    wp.synchronize()

    dev = model.device
    nw = model.world_count

    # Phase 1: step_dt end-to-end
    step_dt_times = []
    k_counts = []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        wp.synchronize()
        step_dt_times.append(time.perf_counter() - t0)
        k_counts.append(int(solver.iteration_count.numpy()[0]))

    # Phase 2: per-component timing (may corrupt solver state)
    timings = {name: [] for name in ALL_COMPONENTS}

    for _ in range(steps):
        effective_dt_max = min(solver._dt_max, DT_OUTER)
        wp.launch(
            _apply_dt_cap, dim=nw,
            inputs=[solver._ideal_dt, solver._dt_min, effective_dt_max, solver._dt, solver._dt_half],
            device=dev,
        )

        # Pre: state copy in
        wp.synchronize()
        t = time.perf_counter()
        wp.copy(solver._state_cur.joint_q, state_0.joint_q)
        wp.copy(solver._state_cur.joint_qd, state_0.joint_qd)
        if state_0.body_q is not None:
            wp.copy(solver._state_cur.body_q, state_0.body_q)
        if state_0.body_qd is not None:
            wp.copy(solver._state_cur.body_qd, state_0.body_qd)
        wp.synchronize()
        t_new = time.perf_counter()
        timings["pre_state_copy_in"].append(t_new - t)
        t = t_new

        solver._apply_mjc_control(model, state_0, control, solver.mjw_data)
        wp.synchronize()
        t_new = time.perf_counter()
        timings["apply_mjc_control"].append(t_new - t)
        t = t_new

        solver._enable_rne_postconstraint(solver._state_cur)
        wp.launch(_boundary_advance, dim=nw, inputs=[solver._next_time, DT_OUTER], device=dev)
        wp.synchronize()

        # Snapshot copies
        t = time.perf_counter()
        wp.copy(solver._state_saved.joint_q, solver._state_cur.joint_q)
        wp.copy(solver._state_saved.joint_qd, solver._state_cur.joint_qd)
        if solver._state_cur.body_q is not None:
            wp.copy(solver._state_saved.body_q, solver._state_cur.body_q)
        if solver._state_cur.body_qd is not None:
            wp.copy(solver._state_saved.body_qd, solver._state_cur.body_qd)
        wp.synchronize()
        t_new = time.perf_counter()
        timings["snapshot_copies"].append(t_new - t)
        t = t_new

        # 3 substeps
        for idx, (src_state, dst_state, dt_arr) in enumerate([
            (solver._state_cur, solver._scratch_full, solver._dt),
            (solver._state_cur, solver._scratch_mid, solver._dt_half),
            (solver._scratch_mid, solver._scratch_double, solver._dt_half),
        ], start=1):
            solver._update_mjc_data(solver.mjw_data, model, src_state)
            wp.synchronize()
            t_new = time.perf_counter()
            timings[f"substep{idx}_update_mjc"].append(t_new - t)
            t = t_new

            wp.copy(solver.mjw_model.opt.timestep, dt_arr)
            wp.synchronize()
            t_new = time.perf_counter()
            timings[f"substep{idx}_dt_copy"].append(t_new - t)
            t = t_new

            with wp.ScopedDevice(dev):
                solver._mujoco_warp_step()
            wp.synchronize()
            t_new = time.perf_counter()
            timings[f"substep{idx}_mujoco"].append(t_new - t)
            t = t_new

            solver._update_newton_state(model, dst_state, solver.mjw_data)
            wp.synchronize()
            t_new = time.perf_counter()
            timings[f"substep{idx}_update_newton"].append(t_new - t)
            t = t_new

        # Error control
        wp.launch(
            _inf_norm_state_error_kernel, dim=nw,
            inputs=[
                solver._scratch_full.joint_q, solver._scratch_double.joint_q,
                solver._q_weights, solver._coords_per_world,
            ],
            outputs=[solver._last_error], device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["inf_norm_error"].append(t_new - t)
        t = t_new

        wp.launch(
            _calc_adjusted_step, dim=nw,
            inputs=[solver._last_error, solver._dt, solver._ideal_dt, solver._accepted, solver._tol, solver._dt_min],
            device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["calc_adjusted_step"].append(t_new - t)
        t = t_new

        # State selection
        wp.launch(
            _select_float_kernel, dim=model.joint_coord_count,
            inputs=[solver._scratch_double.joint_q, solver._state_saved.joint_q, solver._accepted, solver._coords_per_world],
            outputs=[solver._state_cur.joint_q], device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["select_joint_q"].append(t_new - t)
        t = t_new

        wp.launch(
            _select_float_kernel, dim=model.joint_dof_count,
            inputs=[solver._scratch_double.joint_qd, solver._state_saved.joint_qd, solver._accepted, solver._dofs_per_world],
            outputs=[solver._state_cur.joint_qd], device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["select_joint_qd"].append(t_new - t)
        t = t_new

        if solver._state_cur.body_q is not None:
            wp.launch(
                _select_transform_kernel, dim=model.body_count,
                inputs=[solver._scratch_double.body_q, solver._state_saved.body_q, solver._accepted, solver._bodies_per_world],
                outputs=[solver._state_cur.body_q], device=dev,
            )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["select_body_q"].append(t_new - t)
        t = t_new

        if solver._state_cur.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel, dim=model.body_count,
                inputs=[solver._scratch_double.body_qd, solver._state_saved.body_qd, solver._accepted, solver._bodies_per_world],
                outputs=[solver._state_cur.body_qd], device=dev,
            )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["select_body_qd"].append(t_new - t)
        t = t_new

        # Bookkeeping
        wp.launch(
            _advance_sim_time, dim=nw,
            inputs=[solver._sim_time, solver._dt, solver._accepted, solver._last_error, solver._accepted_error],
            device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["advance_sim_time"].append(t_new - t)
        t = t_new

        wp.launch(
            _apply_dt_cap, dim=nw,
            inputs=[solver._ideal_dt, solver._dt_min, DT_OUTER, solver._dt, solver._dt_half],
            device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["apply_dt_cap"].append(t_new - t)
        t = t_new

        wp.launch(_boundary_reset, dim=1, inputs=[solver._boundary_flag], device=dev)
        wp.launch(
            _boundary_check, dim=nw,
            inputs=[solver._sim_time, solver._next_time, solver._boundary_flag], device=dev,
        )
        wp.synchronize()
        t_new = time.perf_counter()
        timings["boundary_check"].append(t_new - t)
        t = t_new

        # Post: state copy out
        wp.copy(state_0.joint_q, solver._state_cur.joint_q)
        wp.copy(state_0.joint_qd, solver._state_cur.joint_qd)
        if state_0.body_q is not None:
            wp.copy(state_0.body_q, solver._state_cur.body_q)
        if state_0.body_qd is not None:
            wp.copy(state_0.body_qd, solver._state_cur.body_qd)
        wp.synchronize()
        t_new = time.perf_counter()
        timings["post_state_copy_out"].append(t_new - t)

    # Assemble results
    k_arr = np.array(k_counts)
    result = {
        "step_dt": float(np.mean(step_dt_times)),
        "k_mean": float(np.mean(k_arr)),
        "k_max": int(np.max(k_arr)),
    }
    for name, vals in timings.items():
        result[name] = float(np.mean(vals))
    for group_name, members in PLOT_GROUPS.items():
        result[group_name] = sum(result[m] for m in members)
    result["iter_body_sum"] = sum(result[c] for c in ITER_COMPONENTS)

    print(
        f"  N={n:>5}  step_dt={result['step_dt'] * 1e3:7.2f} ms  "
        f"iter_body={result['iter_body_sum'] * 1e3:7.2f} ms  "
        f"pre_post={result['pre_post'] * 1e3:6.2f} ms  "
        f"K={result['k_mean']:.1f}",
        flush=True,
    )
    return result


def run(ns: list[int], steps: int, warmup: int) -> dict:
    """Run component breakdown at all N values."""
    print(f"Component breakdown  Ns={ns}  steps={steps}  warmup={warmup}", flush=True)
    all_results = []
    for n in ns:
        r = _measure_n(n, steps, warmup)
        all_results.append(r)
    return {"ns": ns, "steps": steps, "warmup": warmup, "results": all_results}


def plot(data: dict, out_dir: Path) -> None:
    """Generate component breakdown plot."""
    ns = data["ns"]
    all_results = data["results"]
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("step_dt", "step_dt (end-to-end)", "black", "-", 2.5),
        ("iter_body_sum", "iter body sum (1 iter)", "grey", "-", 2.0),
        ("mujoco_warp_x3", "mujoco_warp x3", "tab:blue", "-", 1.8),
        ("update_mjc_x3", "update_mjc_data x3", "tab:green", "--", 1.4),
        ("update_newton_x3", "update_newton_state x3", "tab:red", "--", 1.4),
        ("snapshot_copies", "snapshot copies x4", "tab:orange", ":", 1.2),
        ("state_select_x4", "state select x4", "tab:pink", "-.", 1.2),
        ("error_control", "error control", "tab:brown", "-.", 1.2),
        ("dt_copy_x3", "dt copy x3", "tab:cyan", ":", 1.0),
        ("bookkeeping", "bookkeeping", "tab:olive", ":", 1.0),
        ("pre_post", "pre/post overhead", "lime", ":", 1.0),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    for key, label, color, ls, lw in plot_specs:
        ys = [r[key] * 1e3 for r in all_results]
        exp = power_law_exponent(ns, [r[key] for r in all_results])
        ax.plot(
            ns, ys,
            label=f"{label} (N^{exp:.2f})",
            color=color, linestyle=ls, linewidth=lw,
            marker="o", markersize=3,
        )
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time [ms]")
    ax.set_title("CENIC component scaling (log-log)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "components_breakdown.png")

    # Summary table
    print("\n=== Per-component scaling (mean ms per single iteration) ===")
    header = f"{'component':>25}  {'N=' + str(ns[0]):>10}  {'N=' + str(ns[-1]):>10}  {'exponent':>8}"
    print(header)
    print("-" * len(header))
    all_names = ALL_COMPONENTS + list(PLOT_GROUPS.keys()) + ["iter_body_sum", "step_dt"]
    for name in all_names:
        times = [r[name] for r in all_results]
        exp = power_law_exponent(ns, times)
        t_first = times[0] * 1e3
        t_last = times[-1] * 1e3
        flag = " <<<" if exp > 0.05 else ""
        print(f"{name:>25}  {t_first:10.4f}  {t_last:10.4f}  {exp:8.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Per-kernel component breakdown")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data = run(sorted(args.ns), args.steps, args.warmup)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "components.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'components.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
