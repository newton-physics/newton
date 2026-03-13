"""CENIC scaling diagnostic: GPU wall time per component vs N worlds.

Each component is timed with wp.synchronize() bookends to measure true
GPU cost rather than kernel-launch latency.

Usage:
    uv run python scripts/testing/contact/cenic_scaling_diag.py
    uv run python scripts/testing/contact/cenic_scaling_diag.py --ns 1 2 4 8 16 32 64 --steps 100
"""

import argparse
import time

import numpy as np
import warp as wp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.testing.contact.cenic_contact_objects import DT_OUTER, build_model, make_solver


def _sync_time(fn):
    """Run fn(), synchronize, return wall seconds."""
    t0 = time.perf_counter()
    fn()
    wp.synchronize()
    return time.perf_counter() - t0


def measure_n(n: int, steps: int, warmup: int) -> dict[str, float]:
    """Build model with N worlds, warm up, then time each sub-component.

    Returns a dict mapping component name → mean seconds per step_dt call.
    """
    model  = build_model(n)
    solver = make_solver(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    device = model.device

    for _ in range(warmup):
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        wp.synchronize()

    t_step_dt = []
    for _ in range(steps):
        t = _sync_time(lambda: solver.step_dt(DT_OUTER, state_0, state_1, control))
        state_0, state_1 = state_1, state_0
        t_step_dt.append(t)

    def _time_substep_full():
        solver._run_substep(state_0, solver._scratch_full, None, solver._dt)
        solver._run_substep(state_0, solver._scratch_mid, None, solver._dt_half)
        solver._run_substep(solver._scratch_mid, solver._scratch_double, None, solver._dt_half)

    def _time_graph_replay():
        # Time the full CUDA graph replay (error + select + advance — the fused inner step)
        wp.capture_launch(solver._graph)

    def _time_update_mjc():
        solver._update_mjc_data(solver.mjw_data, model, state_0)

    def _time_update_newton():
        solver._update_newton_state(model, state_1, solver.mjw_data)

    def _time_mujoco_warp():
        solver._mujoco_warp_step()

    def _time_boundary_numpy():
        _ = solver._boundary_flag.numpy()[0]

    def _time_status_summary():
        _ = solver.get_status_summary()

    components = {
        "3x_substep":      _time_substep_full,
        "graph_replay":    _time_graph_replay,
        "update_mjc_data": _time_update_mjc,
        "update_newton":   _time_update_newton,
        "mujoco_warp":     _time_mujoco_warp,
        "boundary_numpy":  _time_boundary_numpy,
        "status_summary":  _time_status_summary,
    }

    timings = {k: [] for k in components}
    for _ in range(steps):
        for name, fn in components.items():
            timings[name].append(_sync_time(fn))

    result = {"step_dt": float(np.mean(t_step_dt))}
    for name, vals in timings.items():
        result[name] = float(np.mean(vals))

    SUBSTEPS   = 3
    bodies_per_world = model.body_count // n
    dofs_per_world   = model.joint_dof_count // n
    threads_per_world = (bodies_per_world + dofs_per_world) * 100 * SUBSTEPS
    result["threads_total"] = float(n * threads_per_world)
    result["threads_per_world"] = float(threads_per_world)

    print(
        f"  N={n:>5}  step_dt={result['step_dt']*1e3:6.2f} ms  "
        f"mujoco_warp={result['mujoco_warp']*1e3:5.2f} ms  "
        f"threads_total={result['threads_total']/1e6:.2f}M",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="CENIC GPU scaling diagnostic")
    parser.add_argument(
        "--ns", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="List of world counts to benchmark",
    )
    parser.add_argument("--steps",  type=int, default=50,  help="Timed steps per N")
    parser.add_argument("--warmup", type=int, default=20,  help="Warm-up steps per N")
    parser.add_argument("--out",    type=str, default="cenic_scaling_diag.png", help="Output plot path")
    args = parser.parse_args()

    ns = sorted(args.ns)
    print(f"CENIC scaling diagnostic  Ns={ns}  steps={args.steps}  warmup={args.warmup}", flush=True)

    all_results: list[dict[str, float]] = []
    for n in ns:
        r = measure_n(n, args.steps, args.warmup)
        all_results.append(r)

    components = [
        ("step_dt",         "step_dt (end-to-end)",              "black",      "-",  2.5),
        ("3x_substep",      "3× MuJoCo substep (uncaptured)",    "tab:blue",   "-",  1.5),
        ("mujoco_warp",     "mujoco_warp_step (×1)",             "tab:cyan",   "--", 1.2),
        ("graph_replay",    "CUDA graph replay (full inner step)","tab:orange", "-",  1.5),
        ("update_mjc_data", "update_mjc_data (×1)",              "tab:green",  "--", 1.2),
        ("update_newton",   "update_newton_state (×1)",          "tab:red",    "--", 1.2),
        ("boundary_numpy",  "boundary_flag.numpy() (×1)",        "tab:purple", ":",  1.2),
        ("status_summary",  "get_status_summary() (×1)",         "tab:brown",  ":",  1.2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (ax, yscale) in enumerate(zip(axes, ["linear", "log"])):
        for key, label, color, ls, lw in components:
            ys = [r[key] * 1e3 for r in all_results]
            ax.plot(ns, ys, label=label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=4)

        ax.set_xlabel("N worlds")
        ax.set_ylabel("Wall time per step_dt call [ms]")
        ax.set_title(f"CENIC component scaling ({yscale} y-axis)")
        ax.set_yscale(yscale)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nPlot saved → {args.out}", flush=True)

    threads_total = [r["threads_total"] / 1e6 for r in all_results]
    wall_times    = [r["mujoco_warp"] * 1e3 for r in all_results]

    device = wp.get_device()
    gpu_name = device.name
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        sm_count  = pynvml.nvmlDeviceGetNumGpuCores(h)
        gpu_label = f"{gpu_name}  ({sm_count} SMs)"
    except Exception:
        sm_count  = None
        gpu_label = gpu_name

    thread_out = args.out.replace(".png", "_threads.png")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_threads = "tab:blue"
    color_wall    = "tab:red"

    ax1.set_xlabel("N worlds")
    ax1.set_ylabel("Theoretical threads dispatched (M)", color=color_threads)
    ax1.plot(ns, threads_total, color=color_threads, marker="o", markersize=4,
             linewidth=1.8, label="threads (M)")
    ax1.tick_params(axis="y", labelcolor=color_threads)
    ax1.set_xscale("log", base=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("mujoco_warp wall time [ms]", color=color_wall)
    ax2.plot(ns, wall_times, color=color_wall, marker="s", markersize=4,
             linewidth=1.8, linestyle="--", label="wall time (ms)")
    ax2.tick_params(axis="y", labelcolor=color_wall)

    min_wall = min(wall_times)
    knee_ns = [n for n, w in zip(ns, wall_times) if w > min_wall * 1.2]
    if knee_ns:
        ax1.axvline(knee_ns[0], color="grey", linestyle=":", linewidth=1.2,
                    label=f"saturation ≈ N={knee_ns[0]}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title(f"Thread utilization vs GPU wall time\n{gpu_label}")
    ax1.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(thread_out, dpi=150)
    plt.close(fig)
    print(f"Thread plot saved → {thread_out}", flush=True)

    print("\n=== Mean wall time per step_dt [ms] ===")
    header = f"{'N':>6}" + "".join(f"  {k:>16}" for k in ["step_dt", "3x_substep", "mujoco_warp", "graph_replay", "update_mjc_data", "update_newton", "boundary_numpy", "status_summary"])
    print(header)
    for n, r in zip(ns, all_results):
        row = f"{n:>6}"
        for k in ["step_dt", "3x_substep", "mujoco_warp", "graph_replay", "update_mjc_data", "update_newton", "boundary_numpy", "status_summary"]:
            row += f"  {r[k]*1e3:>16.3f}"
        print(row)


if __name__ == "__main__":
    main()
