# Benchmark platform and scripts/ reorganization

## Problem

The `scripts/` directory has grown organically. Scene definitions, runnable demos, benchmarks, and stale diagnostics are mixed together. `cenic_contact_objects.py` does double duty as both a demo and a shared library. Benchmark results are not versioned. There is no single way to run all benchmarks and compare across commits.

## Goals

1. Clean separation between scenes (data), demos (interactive), benchmarks (measurement), and archive (stale).
2. Single entry point (`uv run -m scripts.bench`) that runs all benchmarks and saves version-keyed results.
3. Each benchmark is its own script, independently runnable for debugging.
4. Fresh-solver isolation per (N, mode) measurement -- the methodology fix from the 2026-03-28 state contamination discovery.
5. Consistent log-log plotting with IQR bands, power-law exponents in legends, and shared style definitions.
6. Results accumulate across git commits so you can compare scaling regressions over time.

## Directory structure

```
scripts/
  scenes/                              # shared scene definitions (no main())
    __init__.py
    contact_objects.py                 #   build_template, build_model, make_solver, constants

  demos/                               # runnable visualizations
    contact_objects.py                 #   interactive demo, imports from scenes/
    step_quadruped.py                  #   quadruped CENIC demo
    variable_step_pendulum.py          #   variable-step double pendulum

  control/                             # control demos (unchanged)
    double_pendulum_lqr.py
    step_anymal_walk.py

  bench/                               # benchmark platform
    __main__.py                        #   CLI entry point
    runner.py                          #   orchestrator: discover, run, collect, plot
    infra.py                           #   fresh-solver helpers, timing, power-law fit
    plotting.py                        #   shared matplotlib utilities
    benchmarks/                        #   one module per benchmark suite
      __init__.py
      scaling.py                       #   N-scaling: graph/loop/fixed/manual
      components.py                    #   per-kernel component breakdown vs N
      accuracy.py                      #   wall time vs tol, error traces, dt traces
    results/                           #   version-organized output (gitignored)
      <git-short-hash>/
        meta.json
        scaling.json
        components.json
        accuracy.json
        plots/
          scaling_wall_time.png
          scaling_iterations.png
          scaling_per_iter.png
          scaling_amortization.png
          components_breakdown.png
          accuracy_wall_vs_tol.png
          accuracy_error_trace.png
          accuracy_dt_trace.png

  archive/                             # superseded diagnostics
    scaling_diag.py
    diag_iteration_spike.py

  ci/                                  # CI scripts (unchanged)
    update_docs_switcher.py
```

## Component details

### scenes/contact_objects.py

Extracted from the current `testing/contact/cenic_contact_objects.py`. Contains only scene construction -- no `main()`, no CLI, no viewer logic.

```python
# Public API
DT_OUTER: float          # 0.01 s (100 Hz)
TOL: float               # 1e-3
DT_INNER_MIN: float      # 1e-6

def build_template() -> newton.ModelBuilder: ...
def build_model(n_worlds: int) -> newton.Model: ...
def make_solver(model, tol=TOL) -> SolverMuJoCoCENIC: ...
def make_fixed_solver(model) -> SolverMuJoCo: ...
```

`make_fixed_solver` is new -- extracts the fixed-step solver construction that currently lives inline in `_measure_fixed`.

### bench/__main__.py

Entry point. Parses CLI args, delegates to `runner.run()`.

```
uv run -m scripts.bench                          # run all benchmarks
uv run -m scripts.bench --only scaling            # run one benchmark
uv run -m scripts.bench --skip accuracy           # skip one
uv run -m scripts.bench --list                    # list available benchmarks
uv run -m scripts.bench --compare abc1234 def5678 # compare two versions
uv run -m scripts.bench --ns 1 4 16 64 256        # override N values
uv run -m scripts.bench --steps 50 --warmup 20    # override timing params
```

### bench/runner.py

Orchestrator. Responsibilities:

1. Resolve current git short hash (7 chars).
2. Collect GPU metadata: device name, driver version, CUDA version, Warp version.
3. Create `results/<hash>/` directory.
4. Discover benchmark modules in `benchmarks/` (any `.py` with a `run()` function).
5. Run each benchmark in a subprocess for GPU state isolation.
6. Each benchmark's `run()` returns a dict; the subprocess prints it as JSON to stdout.
7. Runner captures JSON, saves to `results/<hash>/<benchmark_name>.json`.
8. After all benchmarks, call each benchmark's `plot()` function to generate plots into `results/<hash>/plots/`.
9. Write `meta.json` with commit hash, timestamp, GPU info, CLI args, benchmark durations.

Subprocess protocol: each benchmark script is runnable as `uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256 --steps 50 --warmup 20`. It prints a single JSON object to the last line of stdout. The runner captures that line.

### bench/infra.py

Shared measurement infrastructure. The critical pattern: fresh solver per (N, mode).

```python
def measure(
    build_model_fn: Callable[[int], Model],
    make_solver_fn: Callable[[Model], Solver],
    step_fn: Callable[[Solver, State, State, Control], tuple[State, State]],
    n: int,
    steps: int,
    warmup: int,
    get_k: Callable[[Solver], int] | None = None,
) -> MeasureResult:
    """Fresh model + solver, warmup, timed window. Returns times and K arrays."""
    ...

@dataclass
class MeasureResult:
    times: np.ndarray       # wall seconds per step
    ks: np.ndarray          # iteration counts (ones if not adaptive)
    median: float           # median wall time
    p25: float
    p75: float
    k_mean: float
    k_max: int

def power_law_exponent(ns: list[int], values: list[float]) -> float:
    """Fit log(value) = alpha * log(N) + c; return alpha."""
    ...
```

### bench/plotting.py

Shared plotting utilities. Enforces CLAUDE.md log-log convention.

```python
# Style registry -- consistent colors/markers across all plots
STYLES: dict[str, PlotStyle]   # mode -> {color, marker, linestyle, label}

def log_log_plot(
    ax: Axes,
    ns: list[int],
    series: dict[str, SeriesData],  # mode -> {medians, p25, p75}
    ylabel: str,
    title: str,
    show_exponents: bool = True,
    show_iqr: bool = True,
) -> None:
    """Standard log-log scaling plot with IQR bands and exponents in legend."""
    ...

def save_fig(fig: Figure, path: str, dpi: int = 150) -> None:
    """tight_layout + savefig + close."""
    ...
```

### bench/benchmarks/scaling.py

N-scaling benchmark. This is the cleaned-up version of the current `cenic_kernel_scaling.py` graph-vs-manual mode.

Modes: `graph`, `loop`, `fixed`, `manual`.

Produces 4 plots: wall_time, iterations, per_iter, amortization.

```python
def run(ns, steps, warmup) -> dict:
    """Run all modes, return JSON-serializable results dict."""
    ...

def plot(data: dict, out_dir: Path) -> None:
    """Generate 4 plots from results dict."""
    ...
```

### bench/benchmarks/components.py

Per-kernel component breakdown. This is the cleaned-up version of `cenic_kernel_scaling.py` components mode.

Produces 1 plot: grouped component scaling (log-log) with exponent table.

### bench/benchmarks/accuracy.py

Accuracy and error behavior. Cleaned-up version of what `cenic_benchmark_plots.py` does for figures 1, 4, 5.

Produces 3 plots:
- Wall time vs tolerance (sweep tol at fixed N)
- Inf-norm q error over simulation time
- Adaptive dt trace over simulation time

### meta.json format

```json
{
  "commit": "2d6f995",
  "branch": "main",
  "timestamp": "2026-03-28T14:30:00Z",
  "gpu": "NVIDIA RTX 4090",
  "driver": "570.86.16",
  "cuda": "12.8",
  "warp": "1.7.1",
  "args": {"ns": [1, 4, 16, 64, 256], "steps": 50, "warmup": 20},
  "benchmarks": {
    "scaling": {"status": "ok", "duration_s": 142.3},
    "components": {"status": "ok", "duration_s": 89.1},
    "accuracy": {"status": "ok", "duration_s": 67.8}
  }
}
```

### results/ gitignore

`scripts/bench/results/` is gitignored. Results are local artifacts, not committed. The `--compare` flag works by reading previously saved local results.

## File moves

| Current path | New path | Notes |
|---|---|---|
| `testing/contact/cenic_contact_objects.py` | `scenes/contact_objects.py` (library) + `demos/contact_objects.py` (demo) | Split |
| `testing/contact/cenic_kernel_scaling.py` | `bench/benchmarks/scaling.py` + `bench/benchmarks/components.py` | Split by mode |
| `testing/contact/cenic_benchmark_plots.py` | `bench/benchmarks/accuracy.py` | Rename + refactor |
| `testing/contact/cenic_scaling_diag.py` | `archive/scaling_diag.py` | Archive |
| `testing/contact/diag_iteration_spike.py` | `archive/diag_iteration_spike.py` | Archive |
| `testing/cenic_step_quadruped.py` | `demos/step_quadruped.py` | Move |
| `testing/variable_step_double_pendulum.py` | `demos/variable_step_pendulum.py` | Move |
| `testing/contact/bench_results/` | `bench/results/<hash>/` | Reorganize |

After moves, delete `testing/` directory entirely (including `__pycache__`).

## Import path changes

All imports update from `scripts.testing.contact.cenic_contact_objects` to `scripts.scenes.contact_objects`.

The bench infrastructure imports from `scripts.scenes` for scene definitions and from `scripts.bench.infra` / `scripts.bench.plotting` for shared utilities.

## What is NOT changing

- `scripts/ci/` stays as-is.
- `scripts/control/` stays as-is. These are control demos, not benchmarks.
- The solver code (`solver_mujoco_cenic.py`) is untouched.
- `step_dt_loop` stays on the solver for now (used by the loop benchmark mode).

## Testing the migration

After restructuring:
1. `uv run python scripts/demos/contact_objects.py` runs the interactive demo.
2. `uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20` produces the same 4 scaling plots with the same data as the current `cenic_kernel_scaling.py --mode graph-vs-manual`.
3. `uv run -m scripts.bench` runs all benchmarks and saves results to `results/<hash>/`.
4. Archived scripts are not imported by anything.
