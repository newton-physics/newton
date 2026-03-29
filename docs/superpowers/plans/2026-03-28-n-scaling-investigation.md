# N-Scaling Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the N^0.094 scaling exponent in `capture_while` is fundamental to CUDA conditional graph replay or reducible, and implement the best-performing stepping strategy.

**Architecture:** Three sequential experiments, each building on the previous result. Experiment 1 isolates the CUDA driver overhead with a minimal capture_while graph. Experiment 2 (conditional on Exp 1) tests whether reducing graph node count lowers the exponent. Experiment 3 measures the sync-based approach that was never benchmarked. Each experiment adds a new benchmark module under `scripts/bench/benchmarks/` and a vault note under `claude_context/scaling/`.

**Tech Stack:** Warp (wp.capture_while, wp.ScopedCapture, wp.capture_launch), CUDA graphs, matplotlib

**Key references:**
- Solver: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`
- Bench infra: `scripts/bench/infra.py`, `scripts/bench/plotting.py`
- Scene: `scripts/scenes/contact_objects.py`
- Existing scaling benchmark: `scripts/bench/benchmarks/scaling.py`
- Warp capture_while: `.local/lib/python3.13/site-packages/warp/_src/context.py:7816`
- Vault index: `claude_context/README.md`
- Scaling approaches log: `~/.claude/projects/.../memory/scaling_approaches.md`

**Decision gates:**
- After Task 3 (run Exp 1): If no-op exponent >= 0.08, the scaling is fundamental to CUDA conditional replay. Skip Task 4-5 (Exp 2) and proceed to Task 6 (Exp 3).
- After Task 3: If no-op exponent < 0.04, graph node count is the lever. Proceed to Task 4-5 (Exp 2).
- After Task 7 (run Exp 3): Compare sync-based absolute time vs graph at high N. Pick the winner.

---

### Task 1: Create the capture_while micro-benchmark module

**Files:**
- Create: `scripts/bench/benchmarks/capture_while_isolation.py`

This benchmark isolates capture_while overhead from physics work. It creates a minimal CUDA graph containing only trivial kernels inside a capture_while loop, then measures N-scaling of that graph replay.

The key insight: the CENIC solver's iteration body contains ~20+ kernel launches (3 MuJoCo steps, error computation, dt adjustment, state select, boundary check). If the no-op graph also scales at N^0.09, the scaling is in the CUDA driver's conditional node replay mechanism itself, not in graph complexity.

- [ ] **Step 1: Write the no-op kernel and loop body**

```python
# scripts/bench/benchmarks/capture_while_isolation.py

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
```

- [ ] **Step 2: Verify it runs standalone**

Run:
```bash
uv run python -m scripts.bench.benchmarks.capture_while_isolation --ns 1 4 16 --steps 10 --warmup 5
```
Expected: prints median times for 3 sizes x 3 N values, saves JSON + plot, prints summary table with exponents.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/benchmarks/capture_while_isolation.py
git commit -m "Add capture_while isolation micro-benchmark"
```

---

### Task 2: Register the new benchmark in the runner

**Files:**
- Modify: `scripts/bench/runner.py:28`

- [ ] **Step 1: Add capture_while_isolation to BENCHMARK_NAMES**

In `scripts/bench/runner.py`, change line 28 from:

```python
BENCHMARK_NAMES = ["scaling", "components", "accuracy"]
```

to:

```python
BENCHMARK_NAMES = ["scaling", "capture_while_isolation", "components", "accuracy"]
```

Also update `_run_benchmark_subprocess` to pass `--ns`, `--steps`, `--warmup` for the new benchmark. In `scripts/bench/runner.py`, change the condition on line 74 from:

```python
    if "ns" in args and bench_name in ("scaling", "components"):
```

to:

```python
    if "ns" in args and bench_name in ("scaling", "capture_while_isolation", "components"):
```

And on line 77 from:

```python
    if "steps" in args and bench_name in ("scaling", "components"):
```

to:

```python
    if "steps" in args and bench_name in ("scaling", "capture_while_isolation", "components"):
```

And on line 79 from:

```python
    if "warmup" in args and bench_name in ("scaling", "components"):
```

to:

```python
    if "warmup" in args and bench_name in ("scaling", "capture_while_isolation", "components"):
```

- [ ] **Step 2: Verify it appears in --list**

Run:
```bash
uv run -m scripts.bench --list
```
Expected: shows 4 benchmarks including `capture_while_isolation`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/runner.py
git commit -m "Register capture_while_isolation in bench runner"
```

---

### Task 3: Run Experiment 1 and record results

**Files:**
- Create: `claude_context/scaling/2026-03-28-capture-while-isolation.md`
- Modify: `claude_context/README.md`

This is a measurement task, not a code task. Run the benchmark, record the numbers, and make the gate decision.

- [ ] **Step 1: Run the isolation benchmark**

```bash
uv run -m scripts.bench --only capture_while_isolation --ns 1 4 16 64 256 --steps 50 --warmup 20
```

Record the output: exponents for all 3 graph sizes, absolute times at N=1 and N=256.

- [ ] **Step 2: Write the vault note**

Create `claude_context/scaling/2026-03-28-capture-while-isolation.md`:

```markdown
# capture_while isolation micro-benchmark

**Date:** 2026-03-28
**Area:** scaling
**Status:** (success | partial -- fill after results)
**Commit:** (fill with current hash)
**Tags:** #scaling #capture_while

## Goal
Determine whether the N^0.094 scaling exponent is fundamental to CUDA conditional graph replay or depends on graph node count.

## Approach
Minimal capture_while graphs with no physics -- just no-op kernels. Three graph sizes:
- 1 kernel/iter (absolute minimum)
- 4 kernels/iter
- 16 kernels/iter (comparable to CENIC iteration body)

All run at K=3 iterations (matching CENIC's typical count).

## Results

| N | 1 kernel (ms) | 4 kernels (ms) | 16 kernels (ms) |
|---|--------------|----------------|-----------------|
| 1 | (fill) | (fill) | (fill) |
| 4 | (fill) | (fill) | (fill) |
| 16 | (fill) | (fill) | (fill) |
| 64 | (fill) | (fill) | (fill) |
| 256 | (fill) | (fill) | (fill) |

Exponents:
- 1 kernel/iter: N^(fill)
- 4 kernels/iter: N^(fill)
- 16 kernels/iter: N^(fill)

## Verdict
(Fill based on results. Key question: does exponent increase with kernel count?)

### Gate decision
- If 1-kernel exponent >= 0.08: scaling is fundamental to CUDA conditional replay. Skip Experiment 2, proceed to Experiment 3 (sync-based).
- If 1-kernel exponent < 0.04 AND 16-kernel exponent > 0.06: graph node count is the lever. Proceed to Experiment 2.
- If all exponents are similar (~0.09): scaling is in the conditional node mechanism itself, independent of graph size.
```

- [ ] **Step 3: Update the vault index**

Add a row to the Scaling table in `claude_context/README.md`:

```markdown
| [capture_while isolation](scaling/2026-03-28-capture-while-isolation.md) -- micro-benchmark, no physics | `(hash)` | (status) |
```

- [ ] **Step 4: Commit**

```bash
git add claude_context/scaling/2026-03-28-capture-while-isolation.md claude_context/README.md
git commit -m "Record capture_while isolation benchmark results"
```

- [ ] **Step 5: Make the gate decision**

Read the exponents. Follow the gate decision criteria from the vault note. If the exponent is fundamental (>= 0.08 at 1 kernel), skip Tasks 4-5 and go to Task 6. If reducible (< 0.04 at 1 kernel, increasing with node count), proceed to Task 4.

---

### Task 4: (Conditional) Profile CENIC graph node count

**GATE: Only do this task if Experiment 1 shows exponent < 0.04 for 1 kernel and increases with kernel count.**

**Files:**
- Modify: `scripts/bench/benchmarks/capture_while_isolation.py`

If graph node count is the lever, we need to know how many nodes the CENIC iteration body has and identify fusible sequences.

- [ ] **Step 1: Add a node-count-vs-exponent analysis to the isolation benchmark**

Add a new graph size to `GRAPH_SIZES` in `capture_while_isolation.py`:

```python
GRAPH_SIZES = {
    "noop_1k": 1,
    "small_4k": 4,
    "medium_16k": 16,
    "large_32k": 32,   # above CENIC's ~20 kernels
    "xlarge_64k": 64,  # stress test
}
```

- [ ] **Step 2: Re-run and plot exponent vs kernel count**

```bash
uv run -m scripts.bench --only capture_while_isolation --ns 1 4 16 64 256 --steps 50 --warmup 20
```

If exponent increases linearly with kernel count, the path forward is kernel fusion in the CENIC body. Update the vault note with the correlation.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/benchmarks/capture_while_isolation.py
git commit -m "Add larger graph sizes to isolation benchmark"
```

---

### Task 5: (Conditional) Investigate kernel fusion opportunities in CENIC body

**GATE: Only do this task if Task 4 confirms exponent scales with node count.**

**Files:**
- Read only (analysis, no code changes yet): `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`

This is a research task. Count the kernel launches in `_run_iteration_body()` and identify which sequences could be fused:

- [ ] **Step 1: Catalog all kernel launches in _run_iteration_body**

From reading `solver_mujoco_cenic.py:362-467`, the iteration body launches:
1. `_iter_count_increment` (dim=1)
2. `wp.copy` x 2-4 (state snapshot)
3. `_run_substep` x 3 (each does: `_update_mjc_data` + `wp.copy(timestep)` + `_mujoco_warp_step` + `_update_newton_state`)
4. `_inf_norm_q_error_kernel` (dim=n)
5. `_calc_adjusted_step` (dim=n)
6. `_select_float_kernel` x 2 (joint_q, joint_qd)
7. `_select_transform_kernel` x 1 (body_q)
8. `_select_spatial_vector_kernel` x 1 (body_qd)
9. `_advance_sim_time` (dim=n)
10. `_apply_dt_cap_dev` (dim=n)
11. `_boundary_reset` (dim=1)
12. `_boundary_check` (dim=n)

The `_run_substep` calls are the heaviest -- each invokes `_mujoco_warp_step()` which itself launches many kernels.

Fusion candidates (CENIC-owned kernels only):
- `_select_float_kernel` + `_select_transform_kernel` + `_select_spatial_vector_kernel` could be one kernel with multiple arrays
- `_advance_sim_time` + `_apply_dt_cap_dev` + `_boundary_reset` + `_boundary_check` could be one kernel

Write findings as a comment in the vault note. Do NOT modify the solver code in this task.

- [ ] **Step 2: Update vault note with fusion analysis**

Add a "Fusion analysis" section to the isolation vault note with the kernel count and fusion candidates.

- [ ] **Step 3: Commit**

```bash
git add claude_context/scaling/2026-03-28-capture-while-isolation.md
git commit -m "Add kernel fusion analysis to isolation vault note"
```

---

### Task 6: Add sync-based step_dt benchmark mode

**Files:**
- Modify: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`
- Modify: `scripts/bench/benchmarks/scaling.py`

The sync-based approach (Approach 1, commit c2ec695) was never measured. It uses a CUDA graph for the iteration body but a Python loop for the boundary check (`.numpy()` on the 4-byte flag per iteration). This is different from `step_dt_loop` which launches kernels individually -- sync-based replays a captured graph per iteration.

- [ ] **Step 1: Add step_dt_sync method to the solver**

Add after `step_dt_loop` (line 726) in `solver_mujoco_cenic.py`:

```python
    def step_dt_sync(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Like :meth:`step_dt` but replays a captured iteration-body graph
        per iteration with a Python boundary loop.

        Each iteration replays the captured CUDA graph for
        ``_run_iteration_body``, then checks the boundary flag via a single
        ``.numpy()`` call (4 bytes). This avoids capture_while's conditional
        graph node overhead while keeping per-iteration kernel launch overhead
        near zero.

        Args:
            dt_outer: Outer control/render period [s].
            state_0: Current state (input/output).
            state_1: Scratch state (unused; returned unchanged).
            control: Control inputs (applied once, persists across substeps).
            apply_forces: Optional ``fn(state)`` for external forces.

        Returns:
            ``(state_0, state_1)`` with ``state_0`` updated.
        """
        device = self.model.device
        n = self.model.world_count

        effective_dt_max = min(self._dt_max, dt_outer)
        self._effective_dt_max_buf.fill_(effective_dt_max)

        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=device,
        )

        wp.copy(self._state_cur.joint_q, state_0.joint_q)
        wp.copy(self._state_cur.joint_qd, state_0.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(self._state_cur.body_q, state_0.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(self._state_cur.body_qd, state_0.body_qd)

        self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
        if apply_forces is not None:
            apply_forces(state_0)

        self._enable_rne_postconstraint(self._state_cur)

        wp.launch(_boundary_advance, dim=n,
                  inputs=[self._next_time, dt_outer], device=device)

        # Capture iteration body as a standalone graph (not conditional).
        if self._iter_body_graph is None:
            self._run_iteration_body()  # warmup
            wp.synchronize()
            with wp.ScopedCapture(device=device) as cap:
                self._run_iteration_body()
            self._iter_body_graph = cap.graph

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)

        while True:
            wp.capture_launch(self._iter_body_graph)
            if self._boundary_flag.numpy()[0] == 0:
                break

        wp.copy(state_0.joint_q, self._state_cur.joint_q)
        wp.copy(state_0.joint_qd, self._state_cur.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(state_0.body_q, self._state_cur.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(state_0.body_qd, self._state_cur.body_qd)

        return state_0, state_1
```

Also add `self._iter_body_graph: wp.Graph | None = None` to `__init__`, after `self._graph: wp.Graph | None = None` (line 340):

```python
        self._iter_body_graph: wp.Graph | None = None
```

- [ ] **Step 2: Add "sync" mode to the scaling benchmark**

In `scripts/bench/benchmarks/scaling.py`, add `"sync"` to the MODES list:

```python
MODES = ["graph", "sync", "loop", "fixed", "manual"]
```

Add the sync measurement case in `_measure_mode` (after the `elif mode == "loop":` block):

```python
    elif mode == "sync":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt_sync(DT_OUTER, s0, s1, ctrl)
        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])
        return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)
```

Add a style for "sync" in `scripts/bench/plotting.py`:

```python
    "sync": PlotStyle("#9467bd", "v", "-", "CENIC adaptive (graph + sync)"),
```

- [ ] **Step 3: Run a smoke test**

```bash
uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 --steps 10 --warmup 5
```

Expected: all 5 modes run, sync mode shows timing data.

- [ ] **Step 4: Commit**

```bash
git add newton/_src/solvers/mujoco/solver_mujoco_cenic.py scripts/bench/benchmarks/scaling.py scripts/bench/plotting.py
git commit -m "Add sync-based step_dt and benchmark mode"
```

---

### Task 7: Run Experiment 3 (full scaling benchmark with sync mode)

**Files:**
- Create: `claude_context/scaling/2026-03-28-sync-based-measured.md`
- Modify: `claude_context/README.md`
- Modify: memory file `scaling_approaches.md`

- [ ] **Step 1: Run the full scaling benchmark**

```bash
uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20
```

Record all 5 modes' results.

- [ ] **Step 2: Write the vault note**

Create `claude_context/scaling/2026-03-28-sync-based-measured.md`:

```markdown
# Sync-based step_dt (graph replay + Python boundary loop)

**Date:** 2026-03-28
**Area:** scaling
**Status:** (fill)
**Commit:** (fill)
**Tags:** #scaling

## Goal
Measure the sync-based approach that was abandoned in favor of capture_while without benchmarking (Approach 1). Determine if graph replay + .numpy() boundary check is flat AND fast enough to beat capture_while at high N.

## Approach
`step_dt_sync()`: captures `_run_iteration_body()` as a standalone CUDA graph (not conditional). Replays it in a Python while loop, checking `_boundary_flag.numpy()[0]` (4 bytes, one int32) per iteration.

Differs from `step_dt_loop` (which launches kernels individually) because the iteration body is a graph replay (near-zero CPU launch overhead per kernel within the iteration).

## Results

| N | Graph (ms) | Sync (ms) | Loop (ms) | Fixed (ms) | Manual (ms) | K_graph | K_sync |
|---|-----------|----------|----------|-----------|------------|---------|--------|
| 1 | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) |
| 4 | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) |
| 16 | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) |
| 64 | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) |
| 256 | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) | (fill) |

Exponents:
- Graph (capture_while): N^(fill)
- Sync (graph + Python loop): N^(fill)
- Loop (individual launches): N^(fill)
- Fixed: N^(fill)
- Manual: N^(fill)

## Verdict
(Fill based on results.)

### Performance hierarchy at N=256
(Fill: which is fastest? Does sync beat graph at high N?)

### Crossover analysis
(Fill: at what N does sync become preferable to graph, if ever?)
```

- [ ] **Step 3: Update vault index and scaling approaches memory**

Add to `claude_context/README.md` Scaling table:
```markdown
| [Sync-based measured](scaling/2026-03-28-sync-based-measured.md) -- graph replay + Python boundary loop | `(hash)` | (status) |
```

Add Approach 6 to memory file `scaling_approaches.md`:
```markdown
## Approach 6: Sync-based graph replay (step_dt_sync)

- **Commit:** (fill)
- **What it was:** Iteration body captured as standalone CUDA graph. Replayed in Python while loop with .numpy() boundary check. Graph replay eliminates per-kernel CPU launch overhead; .numpy() adds ~Xms constant overhead per iteration.
- **Scaling:** N^(fill)
- **Absolute:** N=1: (fill)ms, N=256: (fill)ms
- **Verdict:** (fill)
```

- [ ] **Step 4: Commit**

```bash
git add claude_context/scaling/2026-03-28-sync-based-measured.md claude_context/README.md
git commit -m "Record sync-based step_dt benchmark results"
```

---

### Task 8: Final analysis and recommendation

**Files:**
- Create: `claude_context/scaling/2026-03-28-n-scaling-verdict.md`
- Modify: `claude_context/README.md`

- [ ] **Step 1: Write the final verdict note**

Create `claude_context/scaling/2026-03-28-n-scaling-verdict.md` synthesizing all three experiments:

```markdown
# N-scaling investigation verdict

**Date:** 2026-03-28
**Area:** scaling
**Status:** (fill)
**Commit:** (fill)
**Tags:** #scaling #verdict

## Question
Is the N^0.094 scaling exponent in capture_while fundamental to CUDA conditional graph replay, or can it be reduced?

## Evidence

### Experiment 1: capture_while isolation
(Paste exponents for 1/4/16 kernels per iteration)

### Experiment 2: Node count correlation (if done)
(Paste correlation data, or "SKIPPED -- exponent was fundamental")

### Experiment 3: Sync-based alternative
(Paste sync exponent and absolute times)

## Verdict

### Is it fundamental?
(Yes/No, with evidence)

### Recommended stepping strategy
(Which method to use going forward and why)

### Remaining optimization opportunities
(If any)
```

- [ ] **Step 2: Update vault index**

Add to `claude_context/README.md` Scaling table:
```markdown
| [N-scaling verdict](scaling/2026-03-28-n-scaling-verdict.md) -- synthesis of isolation + sync experiments | `(hash)` | (status) |
```

- [ ] **Step 3: Commit**

```bash
git add claude_context/scaling/2026-03-28-n-scaling-verdict.md claude_context/README.md
git commit -m "Record N-scaling investigation verdict"
```
