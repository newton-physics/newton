# Benchmark Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `scripts/` into scenes, demos, bench, and archive directories; build a single-entry-point benchmark platform with fresh-solver isolation and version-keyed results.

**Architecture:** Scene definitions (model building, solver construction) are extracted into `scripts/scenes/`. Demos become thin wrappers that import scenes. The bench platform has a runner that discovers and executes benchmark modules in subprocesses, saving JSON results keyed by git commit hash. Shared measurement (`infra.py`) and plotting (`plotting.py`) modules enforce the fresh-solver-per-mode pattern and consistent log-log styling.

**Tech Stack:** Python 3.10+, matplotlib, numpy, warp, newton

**Spec:** `docs/superpowers/specs/2026-03-28-bench-platform-design.md`

---

### Task 1: Create scenes/contact_objects.py (extract scene library)

**Files:**
- Create: `scripts/scenes/__init__.py`
- Create: `scripts/scenes/contact_objects.py`

- [ ] **Step 1: Create `scripts/scenes/__init__.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Create `scripts/scenes/contact_objects.py`**

Extract the scene-building code from `scripts/testing/contact/cenic_contact_objects.py`. This file has NO `main()`, NO CLI, NO viewer. Only model construction and solver factories.

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact objects scene: 9 spheres + 9 tilted boxes per world.

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01  # 100 Hz control / render cadence [s]
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

SPHERE_RADIUS = 0.050
BOX_HALF = 0.050
GRID_STEP = 0.200
GRID_OFFSETS = [-GRID_STEP, 0.0, GRID_STEP]
Z_SPHERES = 1.00
Z_BOXES = 1.25


def build_template() -> newton.ModelBuilder:
    """Single-world template: 9 spheres + 9 tilted boxes."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    cfg_obj = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005)

    for ox in GRID_OFFSETS:
        for oy in GRID_OFFSETS:
            b = template.add_body(
                xform=wp.transform(p=wp.vec3(ox, oy, Z_SPHERES), q=wp.quat_identity()),
            )
            template.add_shape_sphere(b, radius=SPHERE_RADIUS, cfg=cfg_obj)

    _box_angles = [
        (15, 0, 0),
        (-20, 10, 0),
        (35, 0, 15),
        (0, 25, -10),
        (49, 0, 0),
        (-30, 20, 5),
        (10, -35, 0),
        (0, 15, 40),
        (-15, 0, -25),
    ]
    for (ox, oy), (ax, ay, az) in zip(
        [(ox, oy) for ox in GRID_OFFSETS for oy in GRID_OFFSETS],
        _box_angles,
    ):
        rx, ry, rz = math.radians(ax), math.radians(ay), math.radians(az)
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        q = wp.quat(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
        b = template.add_body(xform=wp.transform(p=wp.vec3(ox, oy, Z_BOXES), q=q))
        template.add_shape_box(b, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_obj)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane + invisible walls."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()

    cfg_wall = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005, is_visible=False)
    half_inner = 0.350
    wt = 0.025
    wh = 0.750
    for px, py, hx, hy in [
        (-(half_inner + wt), 0.0, wt, half_inner + wt),
        (half_inner + wt, 0.0, wt, half_inner + wt),
        (0.0, -(half_inner + wt), half_inner + wt, wt),
        (0.0, half_inner + wt, half_inner + wt, wt),
    ]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(px, py, wh), q=wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=wh,
            cfg=cfg_wall,
        )
    return builder.finalize()


def make_solver(model: newton.Model, tol: float = TOL) -> newton.solvers.SolverMuJoCoCENIC:
    """CENIC solver with canonical contact-demo parameters."""
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        nconmax=128,
        njmax=640,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=128, njmax=640,
    )
```

- [ ] **Step 3: Verify the scene module imports correctly**

Run: `uv run python -c "from scripts.scenes.contact_objects import build_model, make_solver, make_fixed_solver, DT_OUTER; print('ok')"`

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/scenes/__init__.py scripts/scenes/contact_objects.py
git commit -m "Extract contact objects scene into scripts/scenes/"
```

---

### Task 2: Create demos/ directory (move demo scripts)

**Files:**
- Create: `scripts/demos/contact_objects.py`
- Move: `scripts/testing/cenic_step_quadruped.py` -> `scripts/demos/step_quadruped.py`
- Move: `scripts/testing/variable_step_double_pendulum.py` -> `scripts/demos/variable_step_pendulum.py`

- [ ] **Step 1: Create `scripts/demos/contact_objects.py`**

This is the interactive demo, now importing from `scripts.scenes.contact_objects`. Copy the `main()` and `_print_status()` functions from the original `cenic_contact_objects.py`, update the import.

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive contact objects demo.

Run:
    uv run python scripts/demos/contact_objects.py
    uv run python scripts/demos/contact_objects.py --num-worlds 4
    uv run python scripts/demos/contact_objects.py --fixed-dt 0.001
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers

from scripts.scenes.contact_objects import DT_OUTER, LOG_EVERY, build_model, make_solver


_grid_lines = 0


def _print_status(solver, step):
    global _grid_lines

    n = solver.model.world_count

    if n > 4:
        s = solver.get_status_summary()
        lines = [
            f"  step {step}  tol={solver._tol:.1e}  worlds={n}",
            f"  sim_time  [{s['sim_time_min']:.4f}, {s['sim_time_max']:.4f}] s",
            f"  dt        [{s['dt_min']:.6f}, {s['dt_max']:.6f}] s",
            f"  err_max   {s['error_max']:.3e}",
            f"  accepted  {s['accept_count']}/{n}",
        ]
    else:
        sim_times = solver.sim_time.numpy()
        dts = solver.dt.numpy()
        errors = solver.last_error.numpy()
        accepted = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'L2 error':>{col}}{'status':>{col}}"
        lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
        for i in range(len(sim_times)):
            lines.append(
                f"{'world ' + str(i):>{col}}"
                f"{sim_times[i]:>{col}.4f}"
                f"{dts[i]:>{col}.6f}"
                f"{errors[i]:>{col}.3e}"
                f"{'ok' if accepted[i] else 'REJECT':>{col}}"
            )
        lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1, help="parallel worlds")
    parser.add_argument("--num-steps", type=int, default=0, help="0 = run until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-dt", type=float, default=None,
        help="Use fixed-step SolverMuJoCo with this dt instead of CENIC",
    )
    args = parser.parse_args()

    model = build_model(args.num_worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    use_fixed = args.fixed_dt is not None

    if use_fixed:
        solver = newton.solvers.SolverMuJoCo(
            model, separate_worlds=True, nconmax=128, njmax=640,
        )
        contacts = model.contacts()
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step demo: {args.num_worlds} world(s)  solver=SolverMuJoCo  "
            f"dt={args.fixed_dt:.4e}  substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model)
        print(
            f"CENIC contact demo: {args.num_worlds} world(s)  solver=SolverMuJoCoCENIC  "
            f"tol={solver._tol:.1e}  dt_inner_init={solver._dt.numpy()[0]:.4f}  "
            f"dt_inner_max={solver._dt_max:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(
        pos=wp.vec3(1.97, -2.07, 1.07),
        pitch=-22.5,
        yaw=136.3,
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        if use_fixed:
            for _ in range(n_inner):
                state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                state_0, state_1 = state_1, state_0
        else:
            state_0, state_1 = solver.step_dt(
                DT_OUTER,
                state_0,
                state_1,
                control,
                apply_forces=viewer.apply_forces,
            )
        t += DT_OUTER
        step += 1

        if not use_fixed and step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        viewer.render(state_0, t)

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(
        f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps",
        flush=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Move quadruped and pendulum demos**

```bash
cp scripts/testing/cenic_step_quadruped.py scripts/demos/step_quadruped.py
cp scripts/testing/variable_step_double_pendulum.py scripts/demos/variable_step_pendulum.py
```

These scripts don't import from `scripts.testing.*`, so they need no import changes.

- [ ] **Step 3: Verify demos run**

Run: `uv run python scripts/demos/contact_objects.py --headless --num-steps 5`

Expected: Prints CENIC banner, runs 5 steps, prints FPS summary, exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add scripts/demos/
git commit -m "Add demos/ directory with contact objects, quadruped, pendulum"
```

---

### Task 3: Create bench/infra.py (measurement infrastructure)

**Files:**
- Create: `scripts/bench/__init__.py`
- Create: `scripts/bench/infra.py`

- [ ] **Step 1: Create `scripts/bench/__init__.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Create `scripts/bench/infra.py`**

This is the core measurement infrastructure. The critical pattern: fresh model + solver per measurement call.

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark measurement infrastructure.

Provides fresh-solver-per-measurement timing and power-law fitting.
Every measurement builds a new model, solver, and states from scratch
to prevent state contamination between modes or N values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import warp as wp

import newton


@dataclass
class MeasureResult:
    """Timing result from a single (N, mode) measurement."""

    times: np.ndarray
    ks: np.ndarray
    median: float
    p25: float
    p75: float
    k_mean: float
    k_max: int


def measure(
    build_model_fn: Callable[[int], newton.Model],
    step_fn: Callable[
        [newton.Model, newton.State, newton.State, newton.Control],
        tuple[newton.State, newton.State],
    ],
    n: int,
    steps: int,
    warmup: int,
    get_k: Callable[..., int] | None = None,
) -> MeasureResult:
    """Run a benchmark with a fresh model and solver.

    Args:
        build_model_fn: Callable that takes n_worlds and returns a Model.
        step_fn: Callable that takes (model, s0, s1, ctrl) and returns (s0, s1).
            The step_fn is responsible for creating its own solver internally
            on first call (via closure) or receiving it as a bound argument.
        n: Number of worlds.
        steps: Number of timed steps.
        warmup: Number of warmup steps (not timed).
        get_k: Optional callable returning iteration count after each step.
    """
    model = build_model_fn(n)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()

    for _ in range(warmup):
        s0, s1 = step_fn(model, s0, s1, ctrl)
    wp.synchronize()

    times = []
    ks = []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        s0, s1 = step_fn(model, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        ks.append(get_k() if get_k is not None else 1)

    times_arr = np.array(times)
    ks_arr = np.array(ks, dtype=np.int32)

    return MeasureResult(
        times=times_arr,
        ks=ks_arr,
        median=float(np.median(times_arr)),
        p25=float(np.percentile(times_arr, 25)),
        p75=float(np.percentile(times_arr, 75)),
        k_mean=float(np.mean(ks_arr)),
        k_max=int(np.max(ks_arr)),
    )


def power_law_exponent(ns: list[int], values: list[float]) -> float:
    """Fit log(value) = alpha * log(N) + c; return alpha."""
    valid = [(n, v) for n, v in zip(ns, values) if v > 0]
    if len(valid) < 2:
        return float("nan")
    log_n = np.log([v[0] for v in valid])
    log_v = np.log([v[1] for v in valid])
    alpha, _ = np.polyfit(log_n, log_v, 1)
    return float(alpha)
```

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from scripts.bench.infra import measure, power_law_exponent, MeasureResult; print('ok')"`

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/bench/__init__.py scripts/bench/infra.py
git commit -m "Add bench/infra.py with fresh-solver measurement infrastructure"
```

---

### Task 4: Create bench/plotting.py (shared plot utilities)

**Files:**
- Create: `scripts/bench/plotting.py`

- [ ] **Step 1: Create `scripts/bench/plotting.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared plotting utilities for the benchmark platform.

Enforces log-log axes (per CLAUDE.md convention), consistent styling,
IQR bands, and power-law exponent annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts.bench.infra import power_law_exponent


@dataclass
class PlotStyle:
    color: str
    marker: str
    linestyle: str
    label: str


# Consistent style registry for stepping modes.
STYLES: dict[str, PlotStyle] = {
    "graph": PlotStyle("#1f77b4", "o", "-", "CENIC adaptive (capture_while)"),
    "loop": PlotStyle("#2ca02c", "^", "-", "CENIC adaptive (Python loop)"),
    "fixed": PlotStyle("#ff7f0e", "D", "-", "Fixed-step (dt=10 ms)"),
    "manual": PlotStyle("#d62728", "s", "--", "Manual (fixed K, no graph)"),
}


@dataclass
class SeriesData:
    medians: list[float]
    p25: list[float] | None = None
    p75: list[float] | None = None


def log_log_plot(
    ax: Axes,
    ns: list[int],
    series: dict[str, SeriesData],
    ylabel: str,
    title: str,
    show_exponents: bool = True,
    show_iqr: bool = True,
) -> dict[str, float]:
    """Standard log-log scaling plot. Returns {mode: exponent}."""
    exponents = {}
    for mode, sd in series.items():
        style = STYLES.get(mode)
        if style is None:
            continue
        exp = power_law_exponent(ns, sd.medians)
        exponents[mode] = exp
        label = f'{style.label}  $N^{{{exp:.2f}}}$' if show_exponents else style.label
        ax.plot(
            ns, sd.medians,
            color=style.color, marker=style.marker, ls=style.linestyle,
            lw=2, ms=5, label=label,
        )
        if show_iqr and sd.p25 is not None and sd.p75 is not None:
            ax.fill_between(ns, sd.p25, sd.p75, color=style.color, alpha=0.10)

    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    return exponents


def save_fig(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """tight_layout + savefig + close."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  saved -> {path}", flush=True)
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from scripts.bench.plotting import STYLES, log_log_plot, save_fig, SeriesData; print('ok')"`

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/plotting.py
git commit -m "Add bench/plotting.py with shared log-log plot utilities"
```

---

### Task 5: Create bench/benchmarks/scaling.py

**Files:**
- Create: `scripts/bench/benchmarks/__init__.py`
- Create: `scripts/bench/benchmarks/scaling.py`

- [ ] **Step 1: Create `scripts/bench/benchmarks/__init__.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Create `scripts/bench/benchmarks/scaling.py`**

This is the N-scaling benchmark, extracted from `cenic_kernel_scaling.py` graph-vs-manual mode. Uses `infra.measure()` for fresh-solver isolation.

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""N-scaling benchmark: wall time vs world count for graph/loop/fixed/manual.

Standalone:
    uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256

Produces 4 plots: wall_time, iterations, per_iter, amortization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import warp as wp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.infra import MeasureResult, measure, power_law_exponent
from scripts.bench.plotting import STYLES, SeriesData, log_log_plot, save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_fixed_solver, make_solver

MODES = ["graph", "loop", "fixed", "manual"]


def _step_graph(model, s0, s1, ctrl, _state={}):
    """step_dt via capture_while. Solver created on first call."""
    key = id(model)
    if key not in _state:
        _state[key] = make_solver(model)
    solver = _state[key]
    s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    return s0, s1


def _step_loop(model, s0, s1, ctrl, _state={}):
    """step_dt_loop via Python loop. Solver created on first call."""
    key = id(model)
    if key not in _state:
        _state[key] = make_solver(model)
    solver = _state[key]
    s0, s1 = solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
    return s0, s1


def _step_fixed(model, s0, s1, ctrl, _state={}):
    """Fixed-step SolverMuJoCo. Solver + contacts created on first call."""
    key = id(model)
    if key not in _state:
        solver = make_fixed_solver(model)
        contacts = model.contacts()
        _state[key] = (solver, contacts)
    solver, contacts = _state[key]
    s1 = solver.step(s0, s1, ctrl, contacts, DT_OUTER)
    return s1, s0


def _get_solver(model, _state, factory):
    """Get or create solver for a model, caching by model id."""
    key = id(model)
    if key not in _state:
        _state[key] = factory(model)
    return _state[key]


def _measure_mode(mode: str, n: int, steps: int, warmup: int) -> MeasureResult:
    """Measure one mode at one N with a completely fresh solver."""
    if mode == "graph":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt(DT_OUTER, s0, s1, ctrl)
        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])
        return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)

    elif mode == "loop":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])
        return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)

    elif mode == "fixed":
        state_cache = {}
        def step_fn(model, s0, s1, ctrl):
            key = id(model)
            if key not in state_cache:
                state_cache[key] = (make_fixed_solver(model), model.contacts())
            solver, contacts = state_cache[key]
            s1 = solver.step(s0, s1, ctrl, contacts, DT_OUTER)
            return s1, s0
        return measure(build_model, step_fn, n, steps, warmup)

    elif mode == "manual":
        from scripts.bench.benchmarks._manual_step import measure_manual
        return measure_manual(n, steps, warmup)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run(ns: list[int], steps: int, warmup: int) -> dict:
    """Run all modes at all N values. Returns JSON-serializable dict."""
    data: dict = {"ns": ns, "steps": steps, "warmup": warmup, "modes": {}}

    for mode in MODES:
        mode_data: dict = {
            "medians": [], "p25": [], "p75": [],
            "k_means": [], "k_maxes": [],
        }
        for n in ns:
            result = _measure_mode(mode, n, steps, warmup)
            mode_data["medians"].append(result.median)
            mode_data["p25"].append(result.p25)
            mode_data["p75"].append(result.p75)
            mode_data["k_means"].append(result.k_mean)
            mode_data["k_maxes"].append(result.k_max)
            print(
                f"  N={n:>5}  {mode:>7}  "
                f"median={result.median * 1e3:7.2f} ms  "
                f"K_mean={result.k_mean:.1f}  K_max={result.k_max}",
                flush=True,
            )
        data["modes"][mode] = mode_data

    # Compute exponents.
    data["exponents"] = {
        mode: power_law_exponent(ns, data["modes"][mode]["medians"])
        for mode in MODES
    }
    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate 4 scaling plots from results dict."""
    ns = data["ns"]
    modes_data = data["modes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Wall time per outer step vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    series = {}
    for mode in MODES:
        md = modes_data[mode]
        series[mode] = SeriesData(
            medians=[m * 1e3 for m in md["medians"]],
            p25=[m * 1e3 for m in md["p25"]],
            p75=[m * 1e3 for m in md["p75"]],
        )
    log_log_plot(
        ax, ns, series,
        ylabel="Wall time per outer step [ms]",
        title=f"Scaling: wall time vs N  (DT_outer={DT_OUTER * 1e3:.0f} ms, tol=1e-3)",
    )
    save_fig(fig, out_dir / "scaling_wall_time.png")

    # Plot 2: Iteration count K vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ["graph", "loop"]:
        style = STYLES[mode]
        md = modes_data[mode]
        ax.plot(
            ns, md["k_means"], color=style.color, marker=style.marker,
            ls="-", lw=2, ms=5, label=f'{style.label}  $K_{{mean}}$',
        )
        ax.plot(
            ns, md["k_maxes"], color=style.color, marker=style.marker,
            ls="--", lw=1.5, ms=4, alpha=0.6,
            label=f'{style.label}  $K_{{max}}$',
        )
    ax.axhline(1, color="grey", ls=":", lw=1, label="K = 1 (ideal)")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Iterations per step_dt call", fontsize=11)
    ax.set_title("Adaptive iteration count K vs N  (tol=1e-3)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir / "scaling_iterations.png")

    # Plot 3: Per-iteration wall time vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    per_iter_series = {}
    for mode in MODES:
        md = modes_data[mode]
        per_iter = [m / max(k, 1) for m, k in zip(md["medians"], md["k_means"])]
        per_iter_series[mode] = SeriesData(medians=[p * 1e3 for p in per_iter])
    log_log_plot(
        ax, ns, per_iter_series,
        ylabel="Wall time per iteration [ms]",
        title="Per-iteration wall time vs N  (wall_time / K)",
        show_iqr=False,
    )
    save_fig(fig, out_dir / "scaling_per_iter.png")

    # Plot 4: Cost per world vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    amort_series = {}
    for mode in MODES:
        md = modes_data[mode]
        amort = [m / n_val * 1e3 for m, n_val in zip(md["medians"], ns)]
        amort_series[mode] = SeriesData(medians=amort)
    log_log_plot(
        ax, ns, amort_series,
        ylabel="Wall time per world per outer step [ms]",
        title="GPU amortization: cost per world vs N",
        show_exponents=False,
    )
    save_fig(fig, out_dir / "scaling_amortization.png")

    # Summary table
    print(f"\n{'=' * 72}")
    print("SCALING SUMMARY")
    print(f"{'=' * 72}")
    hdr = f"{'mode':>10}  {'exponent':>8}  {'N=1 (ms)':>10}  {'N=' + str(ns[-1]) + ' (ms)':>12}  {'ratio':>6}  {'K_mean':>6}"
    print(hdr)
    print("-" * len(hdr))
    for mode in MODES:
        md = modes_data[mode]
        t1 = md["medians"][0] * 1e3
        tN = md["medians"][-1] * 1e3
        exp = data["exponents"][mode]
        k = md["k_means"][-1]
        print(
            f"{mode:>10}  N^{exp:<6.3f}  {t1:10.2f}  {tN:12.2f}  "
            f"{tN / t1:5.1f}x  {k:6.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="N-scaling benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = run(sorted(args.ns), args.steps, args.warmup)

    with open(out_dir / "scaling.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'scaling.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `scripts/bench/benchmarks/_manual_step.py`**

The manual measurement mode requires direct access to solver internals and kernel imports. Keep it isolated in its own file to avoid polluting the main scaling module.

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual stepping mode for scaling benchmark.

Measures individual kernel launches at fixed K (no CUDA graph).
Isolated because it imports solver internals.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from scripts.bench.infra import MeasureResult
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver
from newton._src.solvers.mujoco.solver_mujoco_cenic import (
    _apply_dt_cap,
    _boundary_advance,
)


def measure_manual(n: int, steps: int, warmup: int) -> MeasureResult:
    """Manual kernel launches (fixed K, no CUDA graph) with fresh solver."""
    model = build_model(n)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # Warmup with step_dt to capture graph + settle contact.
    for _ in range(warmup):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()

    # Measure K from the same simulation window.
    k_samples = []
    for _ in range(10):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        k_samples.append(int(solver.iteration_count.numpy()[0]))
    k_fixed = max(round(float(np.mean(k_samples))), 1)
    wp.synchronize()

    dev = model.device
    nw = model.world_count

    times = []
    for _ in range(steps):
        effective_dt_max = min(solver._dt_max, DT_OUTER)
        solver._effective_dt_max_buf.fill_(effective_dt_max)
        wp.launch(
            _apply_dt_cap, dim=nw,
            inputs=[solver._ideal_dt, solver._dt_min, effective_dt_max,
                    solver._dt, solver._dt_half],
            device=dev,
        )

        wp.synchronize()
        t0 = time.perf_counter()

        wp.copy(solver._state_cur.joint_q, s0.joint_q)
        wp.copy(solver._state_cur.joint_qd, s0.joint_qd)
        if s0.body_q is not None:
            wp.copy(solver._state_cur.body_q, s0.body_q)
        if s0.body_qd is not None:
            wp.copy(solver._state_cur.body_qd, s0.body_qd)
        solver._apply_mjc_control(model, s0, ctrl, solver.mjw_data)
        solver._enable_rne_postconstraint(solver._state_cur)
        wp.launch(_boundary_advance, dim=nw,
                  inputs=[solver._next_time, DT_OUTER], device=dev)

        for _ in range(k_fixed):
            solver._run_iteration_body()

        wp.copy(s0.joint_q, solver._state_cur.joint_q)
        wp.copy(s0.joint_qd, solver._state_cur.joint_qd)
        if s0.body_q is not None:
            wp.copy(s0.body_q, solver._state_cur.body_q)
        if s0.body_qd is not None:
            wp.copy(s0.body_qd, solver._state_cur.body_qd)

        wp.synchronize()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    ks_arr = np.full(steps, k_fixed, dtype=np.int32)

    return MeasureResult(
        times=times_arr,
        ks=ks_arr,
        median=float(np.median(times_arr)),
        p25=float(np.percentile(times_arr, 25)),
        p75=float(np.percentile(times_arr, 75)),
        k_mean=float(k_fixed),
        k_max=k_fixed,
    )
```

- [ ] **Step 4: Verify scaling benchmark runs standalone**

Run: `uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 --steps 5 --warmup 3`

Expected: Prints timing lines for 4 modes x 2 N values, saves scaling.json, generates 4 plots, prints summary table and final JSON line.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/benchmarks/__init__.py scripts/bench/benchmarks/scaling.py scripts/bench/benchmarks/_manual_step.py
git commit -m "Add scaling benchmark with fresh-solver isolation"
```

---

### Task 6: Create bench/benchmarks/components.py

**Files:**
- Create: `scripts/bench/benchmarks/components.py`

- [ ] **Step 1: Create `scripts/bench/benchmarks/components.py`**

Per-kernel component breakdown, extracted from `cenic_kernel_scaling.py` components mode. Uses `wp.synchronize()` barriers to time each kernel individually.

```python
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
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver
from newton._src.solvers.mujoco.solver_mujoco_cenic import (
    _apply_dt_cap,
    _apply_dt_cap_dev,
    _advance_sim_time,
    _boundary_advance,
    _boundary_check,
    _boundary_reset,
    _calc_adjusted_step,
    _inf_norm_q_error_kernel,
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
    model = build_model(n)
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
        solver._effective_dt_max_buf.fill_(effective_dt_max)
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
            _inf_norm_q_error_kernel, dim=nw,
            inputs=[solver._scratch_full.joint_q, solver._scratch_double.joint_q, solver._coords_per_world],
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
            _apply_dt_cap_dev, dim=nw,
            inputs=[solver._ideal_dt, solver._dt_min, solver._effective_dt_max_buf, solver._dt, solver._dt_half],
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
```

- [ ] **Step 2: Verify it runs standalone**

Run: `uv run python -m scripts.bench.benchmarks.components --ns 1 4 --steps 5 --warmup 3`

Expected: Prints per-component timings for N=1 and N=4, saves components.json, generates breakdown plot.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/benchmarks/components.py
git commit -m "Add per-kernel component breakdown benchmark"
```

---

### Task 7: Create bench/benchmarks/accuracy.py

**Files:**
- Create: `scripts/bench/benchmarks/accuracy.py`

- [ ] **Step 1: Create `scripts/bench/benchmarks/accuracy.py`**

Accuracy and error behavior benchmark, extracted from `cenic_benchmark_plots.py`. Uses subprocess isolation per measurement (same proven pattern).

```python
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
```

- [ ] **Step 2: Verify it runs standalone (quick mode)**

Run: `uv run python -m scripts.bench.benchmarks.accuracy --trials 1 --sim-duration 0.5`

Expected: Prints wall-vs-tol measurements (7 tol values), error/dt traces, saves accuracy.json and 3 plots.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench/benchmarks/accuracy.py
git commit -m "Add accuracy benchmark with wall-vs-tol and trace plots"
```

---

### Task 8: Create bench/runner.py and bench/__main__.py (orchestrator)

**Files:**
- Create: `scripts/bench/runner.py`
- Create: `scripts/bench/__main__.py`

- [ ] **Step 1: Create `scripts/bench/runner.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark orchestrator.

Discovers benchmark modules, runs each in a subprocess, saves
version-keyed results to scripts/bench/results/<git-hash>/.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import warp as wp


BENCHMARKS_PKG = "scripts.bench.benchmarks"
RESULTS_ROOT = Path("scripts/bench/results")

# Benchmark modules in execution order.
BENCHMARK_NAMES = ["scaling", "components", "accuracy"]


def _git_short_hash() -> str:
    """Return 7-char git hash of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gpu_info() -> dict:
    """Collect GPU metadata."""
    info = {"device": "unknown", "warp": wp.__version__}
    try:
        dev = wp.get_device("cuda:0")
        info["device"] = dev.name
    except Exception:
        pass
    return info


def _run_benchmark_subprocess(
    bench_name: str,
    args: dict,
    out_dir: Path,
) -> tuple[float, dict | None]:
    """Run a benchmark in a subprocess. Returns (duration_s, data_dict)."""
    # Build CLI args for the benchmark module.
    cmd = [sys.executable, "-m", f"{BENCHMARKS_PKG}.{bench_name}"]
    cmd.extend(["--out-dir", str(out_dir)])

    if "ns" in args and bench_name in ("scaling", "components"):
        for n in args["ns"]:
            cmd.extend(["--ns", str(n)])
    if "steps" in args and bench_name in ("scaling", "components"):
        cmd.extend(["--steps", str(args["steps"])])
    if "warmup" in args and bench_name in ("scaling", "components"):
        cmd.extend(["--warmup", str(args["warmup"])])
    if "trials" in args and bench_name == "accuracy":
        cmd.extend(["--trials", str(args["trials"])])

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running benchmark: {bench_name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output to terminal.
        text=True,
        timeout=1800,  # 30 min max.
    )
    duration = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", flush=True)
        return duration, None

    # Read the JSON data file that the benchmark wrote.
    json_path = out_dir / f"{bench_name}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return duration, data

    return duration, None


def run(
    only: str | None = None,
    skip: list[str] | None = None,
    args: dict | None = None,
) -> Path:
    """Run benchmarks, save results. Returns output directory path."""
    if args is None:
        args = {}
    if skip is None:
        skip = []

    commit = _git_short_hash()
    out_dir = RESULTS_ROOT / commit
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine which benchmarks to run.
    to_run = BENCHMARK_NAMES
    if only:
        to_run = [only]
    to_run = [b for b in to_run if b not in skip]

    print(f"Benchmark run: commit={commit}  benchmarks={to_run}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    # Run each benchmark.
    meta_benchmarks = {}
    for bench_name in to_run:
        duration, data = _run_benchmark_subprocess(bench_name, args, out_dir)
        status = "ok" if data is not None else "failed"
        meta_benchmarks[bench_name] = {
            "status": status,
            "duration_s": round(duration, 1),
        }

        # Generate plots from the saved JSON data.
        if data is not None:
            try:
                mod = importlib.import_module(f"{BENCHMARKS_PKG}.{bench_name}")
                mod.plot(data, plots_dir)
            except Exception as e:
                print(f"  Plot generation failed: {e}", flush=True)

    # Write meta.json.
    meta = {
        "commit": commit,
        "branch": _git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_gpu_info(),
        "args": args,
        "benchmarks": meta_benchmarks,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved -> {out_dir / 'meta.json'}", flush=True)

    # Summary.
    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete. Results in: {out_dir}")
    for name, info in meta_benchmarks.items():
        print(f"  {name}: {info['status']} ({info['duration_s']:.1f}s)")
    print(f"{'=' * 60}", flush=True)

    return out_dir


def list_benchmarks() -> None:
    """Print available benchmarks."""
    print("Available benchmarks:")
    for name in BENCHMARK_NAMES:
        try:
            mod = importlib.import_module(f"{BENCHMARKS_PKG}.{name}")
            doc = (mod.__doc__ or "").strip().split("\n")[0]
        except ImportError:
            doc = "(import failed)"
        print(f"  {name:15s}  {doc}")
```

- [ ] **Step 2: Create `scripts/bench/__main__.py`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CENIC benchmark platform entry point.

Usage:
    uv run -m scripts.bench                          # run all benchmarks
    uv run -m scripts.bench --only scaling            # run one benchmark
    uv run -m scripts.bench --skip accuracy           # skip one
    uv run -m scripts.bench --list                    # list available benchmarks
    uv run -m scripts.bench --ns 1 4 16 64 256        # override N values
    uv run -m scripts.bench --steps 50 --warmup 20    # override timing params
"""

import argparse

from scripts.bench.runner import list_benchmarks, run


def main():
    parser = argparse.ArgumentParser(
        description="CENIC benchmark platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this benchmark")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Skip these benchmarks")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")

    # Benchmark parameter overrides.
    parser.add_argument("--ns", type=int, nargs="+", default=None,
                        help="Override N values for scaling/components")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override timed steps for scaling/components")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Override warmup steps for scaling/components")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override trial count for accuracy")

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    bench_args = {}
    if args.ns is not None:
        bench_args["ns"] = sorted(args.ns)
    if args.steps is not None:
        bench_args["steps"] = args.steps
    if args.warmup is not None:
        bench_args["warmup"] = args.warmup
    if args.trials is not None:
        bench_args["trials"] = args.trials

    run(only=args.only, skip=args.skip, args=bench_args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify `--list` works**

Run: `uv run -m scripts.bench --list`

Expected: Prints 3 benchmarks (scaling, components, accuracy) with one-line descriptions.

- [ ] **Step 4: Verify `--only scaling` runs end-to-end with small params**

Run: `uv run -m scripts.bench --only scaling --ns 1 4 --steps 5 --warmup 3`

Expected: Runs scaling benchmark in subprocess, saves results to `scripts/bench/results/<hash>/`, generates plots, writes meta.json.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/runner.py scripts/bench/__main__.py
git commit -m "Add benchmark runner and CLI entry point"
```

---

### Task 9: Archive stale diagnostics and clean up old directories

**Files:**
- Move: `scripts/testing/contact/cenic_scaling_diag.py` -> `scripts/archive/scaling_diag.py`
- Move: `scripts/testing/contact/diag_iteration_spike.py` -> `scripts/archive/diag_iteration_spike.py`
- Add: `scripts/bench/results/.gitignore`
- Delete: `scripts/testing/` (entire directory)

- [ ] **Step 1: Create archive/ and move stale scripts**

```bash
mkdir -p scripts/archive
cp scripts/testing/contact/cenic_scaling_diag.py scripts/archive/scaling_diag.py
cp scripts/testing/contact/diag_iteration_spike.py scripts/archive/diag_iteration_spike.py
```

- [ ] **Step 2: Create .gitignore for bench results**

Create `scripts/bench/results/.gitignore`:

```
# Benchmark results are local artifacts, not committed.
*
!.gitignore
```

- [ ] **Step 3: Remove old testing/ directory**

```bash
rm -rf scripts/testing/
```

- [ ] **Step 4: Verify no imports reference old paths**

Run: `grep -r "scripts.testing" scripts/ --include="*.py"`

Expected: No output (all old imports have been replaced in earlier tasks).

- [ ] **Step 5: Verify demos still work**

Run: `uv run python scripts/demos/contact_objects.py --headless --num-steps 3`

Expected: Runs 3 steps, prints FPS summary.

- [ ] **Step 6: Commit**

```bash
git add scripts/archive/ scripts/bench/results/.gitignore
git rm -r scripts/testing/
git commit -m "Archive stale diagnostics, remove testing/, gitignore bench results"
```

---

### Task 10: Update CLAUDE.md and documentation references

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md references to old paths**

In `CLAUDE.md`, update any references to old script paths. The key sections to update:

1. In the "CENIC simulation loop pattern" section, no path changes needed (it references solver code, not scripts).
2. In the "N-scaling optimization log" section, the measurement command references `cenic_kernel_scaling.py`. Update to the new path.

Replace:
```
## N-scaling optimization log

See memory file `scaling_approaches.md` for the full log of approaches tried, their implementations, and measured results. **Before proposing a new approach, read that file. Never retry something already listed. Never move on without recording measured exponents.**
```

With:
```
## N-scaling optimization log

See memory file `scaling_approaches.md` for the full log of approaches tried, their implementations, and measured results. **Before proposing a new approach, read that file. Never retry something already listed. Never move on without recording measured exponents.**

Measurement command: `uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20`
```

- [ ] **Step 2: Update memory file measurement command**

In `scaling_approaches.md`, update the measurement command under `## Rules`:

Replace:
```
- Measure with: `cenic_kernel_scaling.py --mode graph-vs-manual --ns 1 4 16 64 256 --steps 50 --warmup 20`
```

With:
```
- Measure with: `uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20`
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md with new benchmark paths"
```

---

### Task 11: End-to-end verification

- [ ] **Step 1: Run full benchmark suite with small params**

Run: `uv run -m scripts.bench --ns 1 4 --steps 5 --warmup 3 --trials 1`

Expected:
- Scaling benchmark runs (4 modes x 2 N values), saves scaling.json
- Components benchmark runs (2 N values), saves components.json
- Accuracy benchmark runs (7 tols + traces), saves accuracy.json
- meta.json written with commit hash, GPU info, benchmark status/duration
- All plots generated in `results/<hash>/plots/`

- [ ] **Step 2: Verify directory structure matches spec**

```bash
find scripts/ -type f -name "*.py" | sort
```

Expected output should show:
```
scripts/archive/diag_iteration_spike.py
scripts/archive/scaling_diag.py
scripts/bench/__init__.py
scripts/bench/__main__.py
scripts/bench/benchmarks/__init__.py
scripts/bench/benchmarks/_manual_step.py
scripts/bench/benchmarks/accuracy.py
scripts/bench/benchmarks/components.py
scripts/bench/benchmarks/scaling.py
scripts/bench/infra.py
scripts/bench/plotting.py
scripts/bench/runner.py
scripts/ci/update_docs_switcher.py
scripts/control/cenic_double_pendulum_lqr.py
scripts/control/cenic_step_anymal_walk.py
scripts/demos/contact_objects.py
scripts/demos/step_quadruped.py
scripts/demos/variable_step_pendulum.py
scripts/scenes/__init__.py
scripts/scenes/contact_objects.py
```

No `scripts/testing/` should exist.

- [ ] **Step 3: Verify standalone benchmark execution**

Run: `uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 --steps 3 --warmup 2`

Expected: Runs independently, prints results and summary table.

- [ ] **Step 4: Final commit (if any fixups needed)**

Only if steps 1-3 revealed issues that needed fixing.
