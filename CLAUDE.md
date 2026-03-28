## Implementation log workflow

All optimization approaches, solver changes, and non-trivial implementation attempts must be documented in the `claude_context/` Obsidian vault before and after testing. This prevents wasted effort retrying failed approaches.

### Before implementing anything

1. Read the vault index (`claude_context/README.md`) and any notes tagged with the relevant area (e.g. `#scaling`, `#solver`, `#contact`).
2. If the approach (or a close variant) has already been tried, do not retry it. Propose a genuinely different approach instead.

### After testing an implementation

Create or update a note in `claude_context/` with this template:

```markdown
# <Short descriptive title>

**Date:** YYYY-MM-DD
**Area:** scaling | solver | contact | viewer | ...
**Status:** success | failed | partial | abandoned
**Commit:** <hash or "reverted">
**Tags:** #<area> #<status>

## Goal
What we were trying to achieve.

## Approach
What was implemented and why this approach was chosen.

## Results
Measured data -- timings, exponents, error norms, or other quantitative evidence.
Never leave this blank. No result = no note.

## Verdict
Why it worked or didn't. What we learned. Constraints for future attempts.
```

File naming: `YYYY-MM-DD-<short-slug>.md` (e.g. `2026-03-28-capture-while-scaling.md`).

### Vault structure

```
claude_context/
  README.md          -- index of all notes, grouped by area
  scaling/           -- N-scaling optimization attempts
  solver/            -- adaptive stepping, error control, CUDA graph changes
  contact/           -- contact model, broad/narrow phase
  viewer/            -- rendering, VBO, sync
```

### Rules

- Every approach gets a note, even failures. Especially failures.
- Notes must include quantitative results. "It felt faster" is not a result.
- The `README.md` index must stay current -- add a one-line entry when creating a note.
- Migrate existing knowledge: the memory file `scaling_approaches.md` content should be ported into individual vault notes.

---

## CENIC simulation loop pattern

All scripts using `SolverMuJoCoCENIC` must use `step_dt` — never reimplement the inner loop manually.

```python
DT = 0.002  # 500 Hz — default control and render period [s]

while viewer.is_running():
    state_0, state_1 = solver.step_dt(
        DT, state_0, state_1, control,
        apply_forces=viewer.apply_forces,
    )
    # control / policy updates go here — once per DT boundary
    t += DT
    viewer.render(state_0, t)  # begin_frame(t) + log_state + end_frame
```

`step_dt` owns the inner loop, the GPU boundary kernels, `clear_forces`, and `apply_forces` ordering. Never call `viewer.begin_frame`, policy updates, or `for _ in range(N)` inside the inner loop.

---

## CRITICAL: Zero device transfers in the hot path

**Every `.numpy()` call on a GPU array is a full CUDA device synchronization.** In the inner physics loop this fires on every substep — thousands of times per frame during dense contact. This is the single most destructive performance pattern in CENIC scripts.

### The rule: `.numpy()` must never appear inside the inner physics loop.

❌ **Wrong** — stalls the GPU on every substep, O(N) data transferred per stall:
```python
while True:
    solver.step(...)
    if np.all(solver.sim_time.numpy() >= next_time):  # FULL GPU SYNC + N floats
        break
```

✅ **Correct** — all logic stays on GPU; exactly one `int32` exits the device, once per DT boundary:
```python
# This is what step_dt does internally — do not reimplement it.
wp.launch(_boundary_reset, dim=1, inputs=[flag])
wp.launch(_boundary_check, dim=n, inputs=[sim_time, next_time, flag])
if flag.numpy()[0]:   # 1 int32, fires once per render frame
    break
```

### Why N worlds do not hurt physics throughput — but do hurt render throughput

MuJoCo Warp batches all N worlds into a single GPU kernel per step. For small N (≤ ~64 worlds of simple geometry) the GPU is not saturated — physics throughput scales sub-linearly with N, approaching free.

The viewer is the bottleneck at large N. `log_state` calls `wp.synchronize()` once per frame to flush the VBO copy. With N worlds doing more GPU work, that sync takes longer. **For data collection at N > 1, always run `--headless`.**

### Acceptable `.numpy()` call-sites (outside the inner loop)

- `_print_status()` — status grid, gated behind `step % LOG_EVERY`
- Startup banners before the loop (e.g. reading `solver._dt.numpy()` once)
- End-of-run summaries (wall time, FPS)

Any `.numpy()` inside `while True: … solver.step(…)` must be rejected in review.

---

## dt parameter rules

`dt_min` must always be strictly less than `dt_init`. If `dt_min >= dt_init`, any rejected step clamps dt *upward*, which is physically wrong and causes oscillation.

```python
# Correct relationship:  dt_min < dt_init <= dt_max
solver = SolverMuJoCoCENIC(
    model,
    dt_init=1e-3,
    dt_min=5e-4,   # floor — must be < dt_init
    dt_max=0.008,
)
```

---

## Viewer sim-time integration

`viewer.render(state, sim_time)` drives the camera and UI from **simulation time**, not wall clock. This prevents camera jumps during dense contact substeps where many physics steps fire between renders. No additional timing logic is needed in scripts — `render()` handles it.

Multi-world scripts (`--num-worlds N`) produce diverging trajectories even from identical initial conditions. This is expected: GPU floating-point reductions are non-associative, causing per-world RMS error estimates to differ by ULP, which eventually leads to different accept/reject decisions and permanently diverging trajectories. Use `--num-worlds 1` for visualization; use `--headless` for data collection.

---

## Plotting conventions

All benchmark and scaling plots must use log-log axes unless the plot is a time series (x-axis is simulation time) or log scale does not make sense for the data. Time series plots use linear x with log y where appropriate (e.g. error traces).

---

## N-scaling optimization log

See memory file `scaling_approaches.md` for the full log of approaches tried, their implementations, and measured results. **Before proposing a new approach, read that file. Never retry something already listed. Never move on without recording measured exponents.**

Measurement command: `uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20`

@AGENTS.md
