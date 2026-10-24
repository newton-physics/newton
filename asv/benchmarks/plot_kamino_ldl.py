# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate the Markdown report, CSV summary, and figures from the LDL sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmark_kamino_ldl import BACKENDS
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

from newton._src.solvers.kamino._src.linalg.factorize.ldl_sparse import _symbolic

COLORS = ("#167d8d", "#e68632", "#7563af", "#606b74")
LABELS = {"h1": "H1", "g1": "G1 (with hands)", "dr_legs": "DR Legs (closed loops)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("asv/results/kamino_ldl"))
    args = parser.parse_args()
    output = args.input
    metadata = json.loads((output / "metadata.json").read_text())
    has_g1_audit = (output / "g1_audit_metadata.json").exists()
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in metadata["cpu"].splitlines() if line.startswith("Model name:")),
        "See metadata.json",
    )
    parameters = metadata["parameters"]
    robots, worlds = parameters["robots"], parameters["worlds"]
    failures = json.loads((output / "failures.json").read_text())
    data = {}
    for robot in robots:
        for world_count in worlds:
            for backend in parameters["backends"]:
                path = output / f"{robot}_{world_count}_{backend}.json"
                if path.exists():
                    data[robot, world_count, backend] = json.loads(path.read_text())
    if failures or len(data) != len(robots) * len(worlds) * len(parameters["backends"]):
        raise RuntimeError("Resolve or explicitly report failed/incomplete cases before generating a success report")
    fixtures = {r: json.loads((output / f"{r}_fixture.json").read_text()) for r in robots}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def plot_times(filename, title, accessor, ylabel):
        fig, axes = plt.subplots(1, len(robots), figsize=(5 * len(robots), 4.2), squeeze=False)
        for ax, robot in zip(axes[0], robots, strict=True):
            for backend, color in zip(BACKENDS, COLORS, strict=True):
                samples = [np.asarray(accessor(data[robot, w, backend])) for w in worlds]
                medians = np.asarray([np.median(s) for s in samples])
                low = medians - [np.min(s) for s in samples]
                high = np.asarray([np.max(s) for s in samples]) - medians
                ax.errorbar(worlds, medians, yerr=[low, high], marker="o", color=color, label=backend, capsize=3)
            ax.set(xscale="log", yscale="log", xlabel="Parallel worlds", ylabel=ylabel, title=LABELS[robot])
            ax.set_xticks(worlds, [str(w) for w in worlds])
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    plot_times(
        "frame_latency.png",
        "DVI simulation: median and full sample range",
        lambda d: d["frame"]["wall_ms"],
        "Wall time per frame (ms)",
    )
    plot_times(
        "factor_latency.png",
        "Frozen robot joint matrix: numeric factorization",
        lambda d: d["micro"]["factor"]["gpu_ms"],
        "CUDA-event time per factorization (ms)",
    )
    plot_times(
        "solve_latency.png",
        "Frozen robot joint matrix: one right-hand side",
        lambda d: d["micro"]["solve"]["gpu_ms"],
        "CUDA-event time per solve (ms)",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(robots))
    for index, (backend, color) in enumerate(zip(BACKENDS, COLORS, strict=True)):
        storage = [data[r, worlds[0], backend]["micro"]["factor_bytes"] / worlds[0] / 1024 for r in robots]
        axes[0].bar(x + (index - 1.5) * 0.2, storage, width=0.2, color=color, label=backend)
    axes[0].set_xticks(x, [LABELS[r] for r in robots])
    axes[0].set(
        ylabel="Stored factor per world (KiB)", title="Numerical factors only (L and, for LDL, D)", yscale="log"
    )
    axes[0].legend(fontsize=8)
    for index, (backend, color) in enumerate(zip(BACKENDS[:2], COLORS[:2], strict=True)):
        fill = [data[r, worlds[0], backend]["micro"]["fill_blocks_per_world"] for r in robots]
        axes[1].bar(x + (index - 0.5) * 0.3, fill, width=0.3, color=color, label=backend)
    axes[1].set_xticks(x, [LABELS[r] for r in robots])
    axes[1].set(ylabel="Additional 6 x 6 blocks", title="Symbolic fill from each LDL ordering")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "storage_and_fill.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(robots), 3, figsize=(10, 3.4 * len(robots)), squeeze=False)
    for row, robot in enumerate(robots):
        fixture = fixtures[robot]
        groups, edges = fixture["rows"], fixture["edges"]
        original = np.eye(len(groups), dtype=np.int32)
        for i, j in edges:
            original[i, j] = original[j, i] = 1
        group_index = {tuple(group): i for i, group in enumerate(groups)}
        for column, ordering in enumerate((None, "meca", "rcm")):
            pattern = original.copy()
            if ordering:
                ordered, filled_columns = _symbolic(groups, edges, ordering)
                permutation = [group_index[tuple(group)] for group in ordered]
                pattern = original[np.ix_(permutation, permutation)].copy()
                for j, rows_filled in enumerate(filled_columns):
                    for i in rows_filled:
                        if pattern[i, j] == 0:
                            pattern[i, j] = pattern[j, i] = 2
            axes[row, column].imshow(pattern, cmap=ListedColormap(["white", COLORS[0], COLORS[1]]), vmin=0, vmax=2)
            axes[row, column].set(
                title=f"{LABELS[robot]}: {ordering.upper() if ordering else 'original graph'}",
                xlabel="Joint block",
                ylabel="Joint block",
            )
            axes[row, column].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[row, column].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Joint graph and symbolic fill: teal = original blocks, orange = added fill")
    fig.tight_layout()
    fig.savefig(output / "ordering_patterns.png", dpi=180)
    plt.close(fig)

    rows = []
    for (robot, world_count, backend), d in data.items():
        baseline = data[robot, world_count, "LLTBRCM"]
        q, qref = np.asarray(d["first_world_body_q"]), np.asarray(baseline["first_world_body_q"])
        position_difference = float(np.linalg.norm(q[:, :3] - qref[:, :3], axis=1).max())
        rows.append(
            {
                "robot": robot,
                "worlds": world_count,
                "backend": backend,
                "frame_wall_ms": d["frame"]["median_wall_ms"],
                "frame_gpu_ms": float(np.median(d["frame"]["gpu_ms"])),
                "factor_gpu_ms": float(np.median(d["micro"]["factor"]["gpu_ms"])),
                "solve_gpu_ms": float(np.median(d["micro"]["solve"]["gpu_ms"])),
                "factor_bytes_per_world": d["micro"]["factor_bytes"] / world_count,
                "relative_residual": d["micro"]["relative_residual"],
                "relative_solution_error": d["micro"]["relative_solution_error"],
                "backward_error_inf": d["micro"]["backward_error_inf"],
                "max_body_position_difference_vs_lltrcm_m": position_difference,
                "max_abs_velocity": d["max_abs_velocity"],
                "validation": d["validation"],
            }
        )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Kamino DVI: block-sparse LDL versus blocked LLT",
        "",
        f"Measured {metadata['timestamp_utc']} on {metadata['gpu'].splitlines()[1]}.",
        "G1 was rerun after correcting contact support and its velocity assertion; "
        "see [G1 rerun metadata](g1_audit_metadata.json)."
        if has_g1_audit
        else "",
        "",
        "This compares the **same Kamino DVI dynamics solver** with four bilateral joint backends. "
        "LDL-MECA and LDL-RCM store unit-lower L and an explicit diagonal D and solve A = L D Lᵀ. "
        "LLTB and LLTBRCM use Cholesky A = L Lᵀ. This is not a comparison between the source repository's "
        "SolverDVI and Kamino, and it does not benchmark ADMM.",
        "",
        "## End-to-end simulation",
        "",
        "Each entry is the median synchronized wall time for one frame, in milliseconds, for the entire batch. "
        "Lower is better. Timings include collision detection and all physics substeps; they exclude rendering, "
        "asset loading, model construction, compilation, capture, and validation downloads.",
        "",
        "| Robot | Worlds | LDL-MECA | LDL-RCM | LLTB | LLTBRCM | Fastest |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    ratios = {b: [] for b in BACKENDS[:2]}
    for robot in robots:
        for w in worlds:
            times = [data[robot, w, b]["frame"]["median_wall_ms"] for b in BACKENDS]
            report.append(
                f"| {LABELS[robot]} | {w} | "
                + " | ".join(f"{t:.3f}" for t in times)
                + f" | {BACKENDS[int(np.argmin(times))]} |"
            )
            for b, values in ratios.items():
                values.append(data[robot, w, b]["frame"]["median_wall_ms"] / min(times[2:]))
    report += ["", "![Full simulation latency](frame_latency.png)", "", "## What the measurements show", ""]
    for b, values in ratios.items():
        report.append(
            f"- **{b}** takes **{min(values):.2f}-{max(values):.2f}x** the time of the faster LLT "
            "backend in the same workload (below 1 means faster)."
        )
    storage_ratios = [
        data[r, w, b]["micro"]["factor_bytes"] / data[r, w, "LLTBRCM"]["micro"]["factor_bytes"]
        for r in robots
        for w in worlds
        for b in BACKENDS[:2]
    ]
    report.append(
        f"- LDL uses **{min(storage_ratios):.1%}-{max(storage_ratios):.1%}** of LLTBRCM's numerical factor storage. "
        "This measures L and D allocations only, not total solver memory."
    )
    report += [
        "- MECA remains the LDL default. A default ordering is not a claim that LDL wins every workload; "
        "both orderings and both LLT alternatives remain selectable.",
        "- Compare backends within each robot. Robots have different timesteps, contact geometry, drive parameters, "
        "and iteration budgets; frame times across robots are not equal-accuracy comparisons.",
        "",
        "## Isolated factorization and solve",
        "",
        "For each robot, extract one normalized bilateral matrix after the first LLTB frame. "
        "Repeat that exact matrix across the batch and use the same seeded random RHS for every backend. "
        "The matrices and joint graphs are saved alongside the report. These measurements exclude physical assembly. "
        "LDL uses pre-gathered native sparse values; LLT uses its native dense input. LLTBRCM retains its normal "
        "numeric permutation and tile bookkeeping, with the reusable ordering warmed before timing. "
        "Factorization includes the configured failed-pivot check/retry path. Solves use already computed factors. "
        "A single-RHS microbenchmark does not include the complete DVI Schur/contact response path.",
        "",
        "![Factorization latency](factor_latency.png)",
        "",
        "![Single RHS solve latency](solve_latency.png)",
        "",
        "## Storage and fill",
        "",
        "This is **stored numerical factor memory**, not peak solver or GPU memory. It includes the full allocated "
        "L array, plus D for LDL, and excludes input matrices, symbolic metadata, RHS buffers, collision storage, "
        "and Schur workspaces. LDL blocks have six rows with identity padding. LLTBRCM has 32-row capacity padding.",
        "",
        "| Robot | Joint rows | Joint blocks | Graph edges | MECA fill | RCM fill | MECA factor KiB | RCM factor KiB | LLTB factor KiB | LLTBRCM factor KiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in robots:
        f = fixtures[r]
        m = [data[r, worlds[0], b]["micro"] for b in BACKENDS]
        report.append(
            f"| {LABELS[r]} | {f['dimension']} | {len(f['rows'])} | {len(f['edges'])} | "
            f"{m[0]['fill_blocks_per_world']} | {m[1]['fill_blocks_per_world']} | "
            + " | ".join(f"{d['factor_bytes'] / worlds[0] / 1024:.2f}" for d in m)
            + " |"
        )
    report += [
        "",
        "![Factor storage and ordering fill](storage_and_fill.png)",
        "",
        "![Joint block ordering and symbolic fill](ordering_patterns.png)",
        "",
        "The ordering diagram shows structural joint blocks and elimination fill, mirrored across the diagonal "
        "for readability. It is not a scalar numerical sparsity plot. MECA minimizes newly created block edges "
        "at each elimination; block RCM uses the source DVI traversal. LLTBRCM's separate scalar/tile ordering "
        "is not the same permutation as LDL-RCM.",
        "",
        "## Numerical and simulation checks",
        "",
        "Worst error over the measured batch sizes:",
        "",
        "| Robot | Backend | Relative residual ‖Ax-b‖₂/‖b‖₂ | Relative error vs NumPy float64 solve | Backward error (infinity norm) | Retried worlds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in robots:
        for b in BACKENDS:
            selected = [d["micro"] for d in data.values() if d["backend"] == b and d["robot"] == r]
            report.append(
                f"| {LABELS[r]} | {b} | {max(d['relative_residual'] for d in selected):.3e} | "
                f"{max(d['relative_solution_error'] for d in selected):.3e} | "
                f"{max(d['backward_error_inf'] for d in selected):.3e} | "
                f"{sum(d['retry_worlds'] for d in selected)} |"
            )
    report += [
        "",
        f"All **{len(data)}** cases passed finite-state checks and each existing example's `test_final()` "
        "after warmup and after every timed sample. H1 additionally runs its post-settling check at those boundaries. "
        "H1/G1 require all bodies above ground and all velocity components below 0.015; "
        "DR Legs' existing DVI final check requires finite poses and velocities only. "
        "These are fixed-target settling/contact workloads, not locomotion policies or manipulation tasks. "
        "In particular, H1 has no balance policy and settles onto the floor; passing its existing check "
        "does not mean it remains standing.",
        "",
        "The first G1 sweep exposed a validation bug: Warp extracted only `max(abs(qd))` from a multiline "
        "predicate and omitted `< 0.015`. A raw velocity of 0.213 in the 256-world LDL-RCM case revealed it. "
        "The predicate is now on one source line, with a regression that rejects a moving body and accepts "
        "a stationary body. Independent NumPy checks also enforce the bound in the benchmark. "
        "Keeping speculative foot contacts, as H1 already does, reduced the worst checkpoint velocity "
        "in a 256-world diagnostic from 0.229 to 0.0024. All twelve G1 timing cases were rerun with this "
        "same setting for every backend. The final tables use only those corrected G1 runs. "
        "See the [untimed diagnostic](g1_contact_diagnostic.json) and "
        "[archived initial G1 timings](g1_before_validation_fix.csv)."
        if has_g1_audit
        else "",
        "",
        "| Robot | Maximum final body-position difference from LLTBRCM (m, first world) | Maximum absolute velocity component | Matrix condition number |",
        "|---|---:|---:|---:|",
    ]
    for r in robots:
        selected = [d for d in rows if d["robot"] == r]
        report.append(
            f"| {LABELS[r]} | {max(d['max_body_position_difference_vs_lltrcm_m'] for d in selected):.3e} | "
            f"{max(d['max_abs_velocity'] for d in selected):.3e} | {fixtures[r]['condition_number']:.3e} |"
        )
    report += [
        "",
        "Final-position differences are diagnostics, not proof of trajectory equivalence. "
        "Velocity components mix linear m/s and angular rad/s. Full first-world final states are in each raw JSON.",
        "",
        "The isolated numerical acceptance criterion is finite solutions, positive LDL pivots, and "
        "normwise backward error below 1e-6, where backward error is "
        "`||Ax-b||inf / (||A||inf ||x||inf + ||b||inf)`. Relative residual and forward error are reported "
        "separately. The initial pilot used a 0.5% forward-error cutoff; DR Legs exceeded it. "
        "Its severely ill-conditioned closed-loop matrix amplifies float32 rounding. Small backward error "
        "therefore does not imply an accurate solution relative to float64. No iterative refinement "
        "or higher-precision factorization is used here; the forward errors above remain a material limitation.",
        "",
        "## Method and reproduction",
        "",
        f"- Host: {cpu_model}; full CPU and GPU details are in metadata.json.",
        f"- Python `{metadata['python'].split()[0]}`, Warp `{metadata['warp']}`, NumPy `{metadata['numpy']}`.",
        f"- Branch `{metadata['branch']}`; base commit `{metadata['commit']}` plus the changes recorded by "
        "the tracked-diff and script SHA-256 values in [metadata.json](metadata.json).",
        f"- Newton assets pinned to `{metadata['assets_ref']}`.",
        f"- {parameters['warmup']} warmup frames, then {parameters['samples']} consecutive samples of "
        f"{parameters['frames']} frames. No reset between timed samples. Every backend starts from the same "
        "example initial conditions. Microbenchmarks use 10 warmup graph launches and 100 launches per sample.",
        "- One fresh process per case, sequential execution, deterministic shuffled order (seed 20261024). "
        "CUDA graphs enabled for every backend; float32 factors. Both synchronized wall-clock and CUDA-event "
        "sample times are saved. Error bars show the full sample range, not confidence intervals. "
        "Clocks are not locked and results are from one GPU; small differences may be run-to-run noise.",
        "- LLTB uses DVI's factor tile 64 and solve block 256. LLTBRCM uses its default tile 32 and "
        "capacity stride; DR Legs retains its example's parallel-factorization setting. LDL uses block size 6. "
        "All use DVI's failed-pivot shift. No per-case backend tuning was performed.",
        "",
        "| Robot | Bodies/world | Substeps/frame | Frame dt (s) |",
        "|---|---:|---:|---:|",
    ]
    for r in robots:
        d = data[r, worlds[0], BACKENDS[0]]
        report.append(f"| {LABELS[r]} | {d['bodies_per_world']} | {d['substeps']} | {d['frame_dt']:.8f} |")
    report += [
        "",
        "From the repository root, in an environment with the project's dev and examples dependencies "
        "and the versions listed above:",
        "",
        "```bash",
        "uv run --no-sync python asv/benchmarks/benchmark_kamino_ldl.py --output asv/results/kamino_ldl",
        "uv run --no-sync python asv/benchmarks/plot_kamino_ldl.py --input asv/results/kamino_ldl",
        "```",
        "",
        "See [summary.csv](summary.csv) for machine-readable aggregates, [metadata.json](metadata.json) for "
        "hardware and configuration, and the per-case JSON files for every timing sample and numerical check. "
        "The separate [validation.md](validation.md) records the unit and example regression runs.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
