# CUDA Graph Capture for 3-Eval Block

**Date:** 2026-03-13
**Area:** solver
**Status:** success
**Commit:** 84e2877 (initial), 9e3ca18 (bug fix)
**Tags:** #solver #success

## Goal
Capture the 3-eval step-doubling block as a CUDA graph to eliminate Python kernel-dispatch overhead. One `wp.capture_launch()` per inner iteration instead of ~18-36 individual `wp.launch()` calls.

## Approach
- `_run_3eval_block()` method captures the entire iteration as a CUDA graph via `wp.ScopedCapture`
- Added `_state_cur` and `_state_saved` as stable internal buffers (pointers baked into graph)
- `_timestep_buf` stable buffer for `opt.timestep` (pointer must be constant across replays)
- `_run_substep()` uses `wp.copy(self.mjw_model.opt.timestep, dt_array)` instead of Python reference assignment (so the copy is captured in the graph)
- Graph captured once on first `step_dt` call
- `_maybe_recapture()` checked mean dt change (1% threshold) -- later identified as a performance bug and removed

## Results
Graph capture working. One `wp.capture_launch()` per inner iteration replaces ~18-36 individual kernel launches.

### Critical bug (9e3ca18)
The CUDA graph was recording bad error values. Root cause: `self.mjw_model.opt.timestep = dt_array` was a Python reference assignment (not captured in graph), so the graph replayed with stale timestep pointers. All three substeps used whatever timestep was set at capture time, not the current adaptive dt.

**Fix:** replaced Python assignment with `wp.copy()` into a stable `_timestep_buf`, which IS captured as a graph node.

### _maybe_recapture bug (removed in c0273b9)
Was reading `_dt.numpy()` every `step_dt()` call and rebuilding the CUDA graph if dt changed >1%. This meant a device-to-host sync on every outer call, partially defeating the purpose of graph capture. Removed entirely -- graph is now captured once and never rebuilt.

## Verdict
CUDA graph capture is essential for reducing per-iteration overhead. The timestep-pointer bug is the canonical "Python assignment vs. GPU capture" trap -- any value that changes between replays must be written via `wp.copy()` or a kernel, not via Python `=`. The `_maybe_recapture` pattern was wrong in principle: if the graph body's behavior depends on dt, that dependency must be through device-side reads, not through re-capturing.
