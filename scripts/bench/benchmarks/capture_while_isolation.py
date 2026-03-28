# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""capture_while isolation benchmark: minimal graph, variable N.

Measures pure CUDA conditional graph replay overhead, isolated from
physics kernels. Compares no-op (1 kernel), small (4 kernels), and
medium (16 kernels) graph bodies at varying N.

Standalone:
    uv run python -m scripts.bench.benchmarks.capture_while_isolation --ns 1 4 16 64 256
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
from scripts.bench.plotting import SeriesData, log_log_plot, save_fig, PlotStyle, STYLES


# -- Kernels -----------------------------------------------------------------

@wp.kernel
def _noop_kernel(data: wp.array(dtype=wp.float32)):
    """Trivial kernel: read + write one element."""
    i = wp.tid()
    data[i] = data[i] + wp.float32(0.0)


@wp.kernel
def _decrement_counter(counter: wp.array(dtype=wp.int32)):
    """Decrement counter; set to 0 when it reaches zero (loop termination)."""
    if counter[0] > 1:
        counter[0] = counter[0] - 1
    else:
        counter[0] = 0


# -- Graph builders ----------------------------------------------------------

def _build_noop_graph(
    n: int,
    kernels_per_iter: int,
    iterations: int,
    device: str = "cuda:0",
) -> tuple[wp.Graph, wp.array, wp.array]:
    """Build a capture_while graph with `kernels_per_iter` no-op kernels.

    The loop runs exactly `iterations` times (controlled by a decrementing
    counter, same pattern as CENIC's boundary flag).

    Args:
        n: Array size (analogous to world count -- scales kernel dim).
        kernels_per_iter: Number of no-op kernels per loop body.
        iterations: Fixed iteration count for the loop.
        device: CUDA device.

    Returns:
        (graph, data_array, counter_array)
    """
    data = wp.zeros(n, dtype=wp.float32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)

    def loop_body():
        for _ in range(kernels_per_iter):
            wp.launch(_noop_kernel, dim=n, inputs=[data], device=device)
        wp.launch(_decrement_counter, dim=1, inputs=[counter], device=device)

    # Warmup: run once outside capture to prime JIT.
    counter.fill_(iterations)
    loop_body()
    wp.synchronize()

    # Capture the graph.
    counter.fill_(iterations)
    with wp.ScopedCapture(device=device) as capture:
        wp.capture_while(counter, while_body=loop_body)
    return capture.graph, data, counter


# -- Measurement --------------------------------------------------------------

GRAPH_SIZES = {
    "noop_1k": 1,     # 1 kernel per iter (minimal)
    "small_4k": 4,    # 4 kernels per iter
    "medium_16k": 16, # 16 kernels per iter (comparable to CENIC body)
}

ITERATIONS = 3  # Match CENIC's typical K=3


def _measure_one(
    n: int,
    kernels_per_iter: int,
    steps: int,
    warmup: int,
) -> dict:
    """Measure one (N, graph_size) configuration."""
    graph, data, counter = _build_noop_graph(
        n, kernels_per_iter, ITERATIONS,
    )

    # Warmup replays.
    for _ in range(warmup):
        counter.fill_(ITERATIONS)
        wp.capture_launch(graph)
    wp.synchronize()

    # Timed replays.
    times = []
    for _ in range(steps):
        counter.fill_(ITERATIONS)
        wp.synchronize()
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "median": float(np.median(times_arr)),
        "p25": float(np.percentile(times_arr, 25)),
        "p75": float(np.percentile(times_arr, 75)),
    }


def run(ns: list[int], steps: int, warmup: int) -> dict:
    """Run all graph sizes at all N values."""
    data: dict = {"ns": ns, "steps": steps, "warmup": warmup,
                  "iterations": ITERATIONS, "sizes": {}}

    for size_name, kpi in GRAPH_SIZES.items():
        size_data: dict = {"kernels_per_iter": kpi,
                           "medians": [], "p25": [], "p75": []}
        for n in ns:
            result = _measure_one(n, kpi, steps, warmup)
            size_data["medians"].append(result["median"])
            size_data["p25"].append(result["p25"])
            size_data["p75"].append(result["p75"])
            print(
                f"  N={n:>5}  {size_name:>12}  "
                f"median={result['median'] * 1e3:7.3f} ms",
                flush=True,
            )
        data["sizes"][size_name] = size_data

    data["exponents"] = {
        name: power_law_exponent(ns, data["sizes"][name]["medians"])
        for name in GRAPH_SIZES
    }
    return data


# -- Plotting -----------------------------------------------------------------

_ISO_STYLES: dict[str, PlotStyle] = {
    "noop_1k": PlotStyle("#1f77b4", "o", "-", "1 kernel/iter (no-op)"),
    "small_4k": PlotStyle("#2ca02c", "^", "-", "4 kernels/iter"),
    "medium_16k": PlotStyle("#ff7f0e", "D", "-", "16 kernels/iter"),
}


def plot(data: dict, out_dir: Path) -> None:
    """Generate isolation benchmark plot."""
    ns = data["ns"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Register temporary styles for log_log_plot compatibility.
    original_styles = dict(STYLES)
    STYLES.clear()
    STYLES.update(_ISO_STYLES)

    fig, ax = plt.subplots(figsize=(10, 6))
    series = {}
    for size_name in GRAPH_SIZES:
        sd = data["sizes"][size_name]
        series[size_name] = SeriesData(
            medians=[m * 1e3 for m in sd["medians"]],
            p25=[m * 1e3 for m in sd["p25"]],
            p75=[m * 1e3 for m in sd["p75"]],
        )
    log_log_plot(
        ax, ns, series,
        ylabel="Wall time per replay [ms]",
        title=f"capture_while isolation: replay overhead vs N  (K={data['iterations']})",
    )
    save_fig(fig, out_dir / "capture_while_isolation.png")

    # Restore original styles.
    STYLES.clear()
    STYLES.update(original_styles)

    # Summary table.
    print(f"\n{'=' * 60}")
    print("CAPTURE_WHILE ISOLATION SUMMARY")
    print(f"{'=' * 60}")
    hdr = f"{'size':>15}  {'exponent':>10}  {'N=1 (ms)':>10}  {'N={} (ms)':>12}".format(ns[-1])
    print(hdr)
    print("-" * len(hdr))
    for name in GRAPH_SIZES:
        sd = data["sizes"][name]
        t1 = sd["medians"][0] * 1e3
        tN = sd["medians"][-1] * 1e3
        exp = data["exponents"][name]
        print(f"{name:>15}  N^{exp:<7.3f}  {t1:10.3f}  {tN:12.3f}")


def main():
    parser = argparse.ArgumentParser(description="capture_while isolation benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup)

    with open(out_dir / "capture_while_isolation.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'capture_while_isolation.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
