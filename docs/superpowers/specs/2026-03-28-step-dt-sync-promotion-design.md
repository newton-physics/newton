# step_dt sync promotion design

## Summary

Replace capture_while-based `step_dt` with the sync-based graph-replay approach as the sole adaptive stepping path. Remove `step_dt_loop` and all capture_while infrastructure. The investigation proved capture_while's conditional graph nodes interact with MuJoCo's complex graph body to produce N^0.099 scaling, while the sync-based approach achieves N^0.047 (close to Warp's fundamental baseline) and wins on absolute time at N >= 64.

## Motivation

- capture_while `step_dt`: N^0.099, 2.00ms at N=1, 3.67ms at N=256
- sync-based `step_dt_sync`: N^0.047, 2.18ms at N=1, 2.79ms at N=256
- Sync wins at N >= 64 on absolute wall time (24% faster at N=256)
- Only 9% slower at N=1
- Simpler architecture: no conditional graph nodes, no CUDA 12.4+ requirement
- `step_dt_loop` (individual kernel launches) is 5x slower than both -- no reason to keep it

## Changes

### solver_mujoco_cenic.py

**Delete these methods:**
- `step_dt` (capture_while version)
- `step_dt_loop` (individual kernel launches version)
- `_capture_graph`
- `_maybe_recapture`

**Delete from `__init__`:**
- `self._graph: wp.Graph | None = None`

**Rename:**
- `step_dt_sync` -> `step_dt`

**Keep unchanged:**
- `self._iter_body_graph` (flat graph for iteration body replay)
- `step()` (non-graph single-step method)
- `_run_iteration_body()` (captured into the flat graph)
- All kernels, all properties, all `__init__` parameters
- `_apply_dt_cap_dev` (used inside `_run_iteration_body` for in-graph dt capping)
- `_boundary_flag` (used by `_run_iteration_body` for loop termination)

**Update docstrings:**
- Class docstring: replace capture_while description with sync-based graph-replay description. Remove CUDA 12.4+ requirement note.
- `step_dt` docstring: adapted from current `step_dt_sync` docstring

### scripts/bench/benchmarks/scaling.py

**Update MODES:**
- From: `["graph", "sync", "loop", "fixed", "manual"]`
- To: `["graph", "fixed", "manual"]`

"graph" now calls `solver.step_dt()` which IS the sync-based path. The name "graph" is kept because it describes the user-facing concept (CUDA graph replay for adaptive stepping).

**Delete:**
- `_step_loop` function
- `elif mode == "loop"` case in `_measure_mode`
- `elif mode == "sync"` case in `_measure_mode`

**Update:**
- `_step_graph` function: calls `solver.step_dt()` (same as before, since sync was promoted to `step_dt`)
- Iteration count plot: remove "loop" from the mode list (`["graph", "sync", "loop"]` -> `["graph"]`)

### scripts/bench/plotting.py

**Remove from STYLES:**
- `"loop"` entry
- `"sync"` entry

**Keep:**
- `"graph"` entry (now represents the sync-based path)
- `"fixed"` entry
- `"manual"` entry

### CLAUDE.md

**Update "CENIC simulation loop pattern" section:**
- No change to the user-facing code pattern (still `solver.step_dt(...)`)

**Update "Zero device transfers in the hot path" section:**
- Remove the "Correct" code example that showed capture_while internals (`_boundary_reset`, `_boundary_check`, `flag.numpy()`)
- Replace with a note that `step_dt` uses graph replay with a single `.numpy()` check per iteration (3 checks at K=3 typical)
- Keep the rule that `.numpy()` must not appear in user scripts' inner loops -- `step_dt` handles it internally

### Files NOT changed

- `scripts/bench/benchmarks/_manual_step.py` -- calls `solver.step_dt()` for warmup, no change
- `scripts/bench/benchmarks/components.py` -- calls `solver.step_dt()`, no change
- `scripts/bench/benchmarks/accuracy.py` -- calls `solver.step_dt()`, no change
- `scripts/demos/contact_objects.py` -- calls `solver.step_dt()`, no change
- `scripts/archive/` -- stale scripts, left as-is
- `claude_context/` vault notes -- historical records, not updated
- `scripts/bench/benchmarks/capture_while_isolation.py` -- standalone micro-benchmark, no solver dependency

## Verification

After the change, run:
```bash
uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256 --steps 50 --warmup 20
```

Expected: 3 modes (graph, fixed, manual). Graph mode should show N^~0.047 (matching the old sync results). Fixed should be N^~0.004. Manual should be N^~0.026.
