# step_dt sync promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace capture_while-based `step_dt` with sync-based graph-replay as the sole adaptive stepping path, cutting the N-scaling exponent from 0.099 to 0.047.

**Architecture:** Delete `step_dt` (capture_while), `step_dt_loop` (individual launches), and capture_while infrastructure. Promote `step_dt_sync` to `step_dt`. Update benchmarks to remove dead modes. Update CLAUDE.md.

**Tech Stack:** Warp (wp.ScopedCapture, wp.capture_launch), Python

---

### Task 1: Clean up the solver -- delete capture_while, promote sync

**Files:**
- Modify: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`

- [ ] **Step 1: Update the class docstring**

Replace lines 251-273:

```python
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The inner boundary
    loop is a ``wp.capture_while`` conditional CUDA graph -- zero CPU syncs
    or Python overhead per iteration, so wall time is flat as N increases
    until GPU saturation.

    Timesteps are managed internally by the error controller.  Set the
    initial value via ``dt_inner_init`` and query current values via
    :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoCENIC(model, tol=1e-3)
        state_0, state_1 = model.state(), model.state()

        while viewer.is_running():
            state_0, state_1 = solver.step_dt(DT, state_0, state_1, control,
                                               apply_forces=viewer.apply_forces)
            viewer.render(state_0, solver.sim_time.numpy().min())
    """
```

with:

```python
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The iteration body
    is captured as a CUDA graph and replayed per iteration, with a Python
    boundary loop checking a 4-byte flag via ``.numpy()`` per iteration.

    Timesteps are managed internally by the error controller.  Set the
    initial value via ``dt_inner_init`` and query current values via
    :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoCENIC(model, tol=1e-3)
        state_0, state_1 = model.state(), model.state()

        while viewer.is_running():
            state_0, state_1 = solver.step_dt(DT, state_0, state_1, control,
                                               apply_forces=viewer.apply_forces)
            viewer.render(state_0, solver.sim_time.numpy().min())
    """
```

- [ ] **Step 2: Remove `self._graph` from `__init__`**

Delete these two lines (339-340):

```python
        # Captured once on first step_dt call.
        self._graph: wp.Graph | None = None
```

And update the comment on the remaining line (341) from:

```python
        self._iter_body_graph: wp.Graph | None = None
```

to:

```python
        # Iteration body graph, captured once on first step_dt call.
        self._iter_body_graph: wp.Graph | None = None
```

- [ ] **Step 3: Delete `_capture_graph` and `_maybe_recapture`**

Delete lines 470-481:

```python
    def _capture_graph(self) -> None:
        """Build the CUDA graph for the conditional boundary loop."""
        self._run_iteration_body()  # warm-up: primes JIT + CUDA allocations

        with wp.ScopedCapture() as capture:
            wp.capture_while(self._boundary_flag, while_body=self._run_iteration_body)
        self._graph = capture.graph

    def _maybe_recapture(self) -> None:
        """Capture the CUDA graph on first use."""
        if self._graph is None:
            self._capture_graph()
```

- [ ] **Step 4: Delete the old `step_dt` (capture_while version)**

Delete the entire `step_dt` method (lines 483-658, the one with `@event_scope` and `@override` decorators that calls `self._maybe_recapture()` and `wp.capture_launch(self._graph)`).

- [ ] **Step 5: Delete `step_dt_loop`**

Delete the entire `step_dt_loop` method (lines 661-726, the one with the `while True: self._run_iteration_body(); if self._boundary_flag.numpy()[0] == 0: break` loop using individual kernel launches).

- [ ] **Step 6: Rename `step_dt_sync` to `step_dt`**

Change `def step_dt_sync(` to `def step_dt(`, add the `@event_scope` and `@override` decorators, and update the docstring:

```python
    @event_scope
    @override
    def step_dt(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Advance all worlds by exactly ``dt_outer`` seconds of simulation time.

        The iteration body is replayed as a captured CUDA graph per iteration,
        with a Python boundary loop checking a 4-byte flag. This provides
        near-zero per-kernel CPU launch overhead with minimal N-scaling.

        Args:
            dt_outer: Outer control/render period [s].
            state_0: Current state (input/output).
            state_1: Scratch state (unused; returned unchanged).
            control: Control inputs (applied once, persists across substeps).
            apply_forces: Optional ``fn(state)`` for external forces.

        Returns:
            ``(state_0, state_1)`` with ``state_0`` updated.
        """
```

The rest of the method body stays exactly the same (the sync implementation).

- [ ] **Step 7: Update the `iteration_count` property docstring**

Change:

```python
        """Iteration count from the most recent ``step_dt`` or ``step_dt_loop``, shape ``[1]``, int32, on device."""
```

to:

```python
        """Iteration count from the most recent ``step_dt``, shape ``[1]``, int32, on device."""
```

- [ ] **Step 8: Verify the solver still runs**

```bash
uv run python -c "
from scripts.scenes.contact_objects import build_model, make_solver, DT_OUTER
model = build_model(2)
solver = make_solver(model)
s0, s1, ctrl = model.state(), model.state(), model.control()
for _ in range(5):
    s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
print(f'sim_time={solver.sim_time.numpy()}, K={solver.iteration_count.numpy()[0]}')
print('OK')
"
```

Expected: prints sim_time values and K, ends with "OK".

---

### Task 2: Clean up the scaling benchmark

**Files:**
- Modify: `scripts/bench/benchmarks/scaling.py`
- Modify: `scripts/bench/plotting.py`

- [ ] **Step 1: Update MODES and remove dead functions**

In `scripts/bench/benchmarks/scaling.py`:

Change line 30 from:
```python
MODES = ["graph", "sync", "loop", "fixed", "manual"]
```
to:
```python
MODES = ["graph", "fixed", "manual"]
```

Update the `_step_graph` docstring (line 34) from:
```python
    """step_dt via capture_while. Solver created on first call."""
```
to:
```python
    """step_dt via CUDA graph replay. Solver created on first call."""
```

Delete the `_step_loop` function (lines 43-50):
```python
def _step_loop(model, s0, s1, ctrl, _state={}):
    """step_dt_loop via Python loop. Solver created on first call."""
    key = id(model)
    if key not in _state:
        _state[key] = make_solver(model)
    solver = _state[key]
    s0, s1 = solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
    return s0, s1
```

- [ ] **Step 2: Remove dead modes from `_measure_mode`**

Delete the `elif mode == "loop":` block (lines 85-93):
```python
    elif mode == "loop":
        solver_cache = {}
        def step_fn(model, s0, s1, ctrl):
            solver = _get_solver(model, solver_cache, make_solver)
            return solver.step_dt_loop(DT_OUTER, s0, s1, ctrl)
        def get_k():
            solver = next(iter(solver_cache.values()))
            return int(solver.iteration_count.numpy()[0])
        return measure(build_model, step_fn, n, steps, warmup, get_k=get_k)
```

Delete the `elif mode == "sync":` block (lines 95-103):
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

- [ ] **Step 3: Update the iteration count plot**

Change line 181 from:
```python
    for mode in ["graph", "sync", "loop"]:
```
to:
```python
    for mode in ["graph"]:
```

- [ ] **Step 4: Remove dead styles from plotting.py**

In `scripts/bench/plotting.py`, update the STYLES dict.

Change the "graph" label from:
```python
    "graph": PlotStyle("#1f77b4", "o", "-", "CENIC adaptive (capture_while)"),
```
to:
```python
    "graph": PlotStyle("#1f77b4", "o", "-", "CENIC adaptive (graph replay)"),
```

Delete these two entries:
```python
    "sync": PlotStyle("#9467bd", "v", "-", "CENIC adaptive (graph + sync)"),
```
```python
    "loop": PlotStyle("#2ca02c", "^", "-", "CENIC adaptive (Python loop)"),
```

- [ ] **Step 5: Smoke test the benchmark**

```bash
uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 --steps 5 --warmup 3
```

Expected: 3 modes (graph, fixed, manual), no errors, no references to "loop" or "sync".

---

### Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the "Zero device transfers" section**

Replace the "Correct" code example (lines 94-101):

```python
✅ **Correct** — all logic stays on GPU; exactly one `int32` exits the device, once per DT boundary:
```python
# This is what step_dt does internally — do not reimplement it.
wp.launch(_boundary_reset, dim=1, inputs=[flag])
wp.launch(_boundary_check, dim=n, inputs=[sim_time, next_time, flag])
if flag.numpy()[0]:   # 1 int32, fires once per render frame
    break
```
```

with:

```
✅ **Correct** — use `step_dt`, which handles the inner loop internally:
```python
# step_dt replays a captured CUDA graph per iteration and checks a
# 4-byte boundary flag via .numpy() -- one int32 per iteration (K~3).
# Do not reimplement this loop manually.
state_0, state_1 = solver.step_dt(DT, state_0, state_1, control)
```
```

---

### Task 4: Run the full benchmark and verify exponents

**Files:** None (measurement only)

- [ ] **Step 1: Run the full scaling benchmark**

```bash
uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256 --steps 50 --warmup 20
```

Expected: 3 modes. Graph should show N^~0.047 (matching old sync results). Fixed ~N^0.004. Manual ~N^0.026.

- [ ] **Step 2: Verify no regressions**

Check that:
- Graph mode absolute times at N=1 are ~2.2ms (matching old sync)
- Graph mode absolute times at N=256 are ~2.8ms (matching old sync)
- K_mean is ~3.1 (unchanged)
- No errors or warnings
