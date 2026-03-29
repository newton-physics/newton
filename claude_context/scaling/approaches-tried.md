---
aliases:
  - scaling approaches
  - optimization log
date: 2026-03-28
status: reference
tags:
  - scaling
  - reference
---

# Scaling approaches tried

Every N-scaling optimization attempt, what it measured, why it worked or didn't. Read before proposing new approaches.

Measurement command: `uv run -m scripts.bench --only scaling --ns 1 2 4 8 16 32 64 128 256 512 1024 2048 --steps 200 --warmup 50`

## 1. CUDA graph capture + graph replay (Mar 13, commit 84e2877)

**Approach:** Capture the 3-eval step-doubling block (~30 kernel launches) as a CUDA graph via `wp.ScopedCapture`. One `wp.capture_launch()` per iteration replaces individual `wp.launch()` calls. Python while loop checks `.numpy()` boundary flag.

**Result:** Graph replay dispatch time scales with total thread blocks. At high N, more blocks per kernel = more dispatch overhead. Measured N^0.243 per-iteration exponent.

**Why abandoned:** Direct `wp.launch()` has constant CPU overhead per launch. Graph replay overhead grows with N. Direct launches measured N^0.174 -- decisively better at scale.

**Bugs found during this work:**
- Timestep pointer bug: `self.mjw_model.opt.timestep = dt_array` is Python reference assignment, not captured in graph. Fix: `wp.copy()` into stable buffer.
- `_maybe_recapture`: was reading `_dt.numpy()` every `step_dt()` call (full device sync every outer call). Removed entirely.

**Key lesson:** Python `=` assignment is not captured in CUDA graphs. Anything that changes between replays must use `wp.copy()` or a kernel write.

## 2. capture_while conditional graph (Mar 27, commit 2d6f995)

**Approach:** Zero CPU synchronization via `wp.capture_while`. GPU drives the boundary loop autonomously as a conditional CUDA graph node. No `.numpy()` calls at all.

**Result:** N^0.094 per-iteration exponent. Better than graph replay (0.243) but the conditional node replay overhead still scales with thread blocks.

**Why abandoned:** Requires CUDA 12.4+. The direct-launch approach (N^0.174 with `.numpy()`) is simpler, more portable, and the `.numpy()` overhead is negligible compared to the GPU work at high N.

**Prerequisite:** Required zero-alloc step (StepWorkspace patch to mujoco_warp) because `wp.capture_while` cannot tolerate allocations in the loop body. This patch is no longer needed or applied.

## 3. Direct wp.launch() + Python boundary loop (Mar 28, CURRENT)

**Approach:** Call `_run_iteration_body()` directly (each internal kernel dispatched via `wp.launch()`). Python while loop checks `_boundary_flag.numpy()[0]` (4-byte read) per iteration to decide when to stop.

**Result:** N^0.174 per-iteration exponent. All three benchmark modes (CENIC, fixed, manual) scale identically at ~N^0.18, confirming CENIC's adaptive machinery adds zero scaling overhead.

**Why it wins:** Each `wp.launch()` has constant CPU dispatch overhead (~0.2ms). The GPU work scales sub-linearly with N (batched kernels). The `.numpy()` boundary check is a 4-byte device-to-host read, constant cost regardless of N.

See [[2026-03-28-sync-based-measured|current architecture]] for full details.

## 4. Benchmark measurement bug (Mar 28, fixed)

**Bug:** The manual benchmark sampled K (iteration count) during free-fall (sim_time 0.20-0.30s, before contact onset at 0.45s), getting k_fixed=1 instead of k_fixed~5. This made manual mode appear 1.5-2x faster than CENIC, creating a false scaling gap.

**Fix:** Increased manual benchmark warmup to `max(warmup, 50)` so K is sampled during the contact phase.

**Before fix:** CENIC N^0.150 vs manual N^0.089 (appeared to be a real gap)
**After fix:** CENIC N^0.174 vs manual N^0.184 (within noise, no gap)

**Key lesson:** Benchmark warmup must reach the contact phase before sampling adaptive parameters. 20 warmup steps is insufficient for this scene (contact at step ~45).

## Summary table

| Approach | Per-iter exponent | Status |
|----------|-------------------|--------|
| Graph replay (ScopedCapture) | N^0.243 | Replaced |
| capture_while (conditional graph) | N^0.094 | Abandoned (CUDA 12.4+, no real benefit) |
| **Direct wp.launch() + Python loop** | **N^0.174** | **Current** |
| All modes (CENIC/fixed/manual) | ~N^0.18 | Base GPU physics scaling |
