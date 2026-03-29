---
aliases:
  - direct-launch step_dt
  - current architecture
date: 2026-03-28
status: current
commit: uncommitted
tags:
  - scaling
  - current
---

# Direct-launch step_dt (wp.launch + Python boundary loop)

> [!success] Current architecture
> This is the active step_dt implementation. Direct `wp.launch()` kernel calls in a Python `while` loop with `.numpy()` boundary check. No CUDA graph capture or replay.

Calls `_run_iteration_body()` directly (each internal kernel dispatched via `wp.launch()`). Python while loop checks `_boundary_flag.numpy()[0]` (4 bytes) per iteration to decide when to stop.

## Evolution

1. `wp.capture_while` conditional graph -- abandoned (CUDA 12.4+ only, no perf benefit)
2. `wp.ScopedCapture` + graph replay + Python loop -- **replaced** (graph replay dispatch overhead scaled with N)
3. Direct `wp.launch()` + Python loop -- **current** (constant CPU overhead per launch, sub-linear GPU scaling)

## Results

### Latest (N=1..256, contact_objects scene, warmup=50, 50 steps, Mar 29)

Benchmark modes renamed: `graph` -> `cenic`, `manual` -> `single_iter`. single_iter times one `_run_iteration_body()` directly (sync-to-sync, no K division).

| N | single_iter (ms) | fixed (ms) | cenic wall (ms) | K_mean |
|---|-------------------|------------|------------------|--------|
| 1 | 7.5 | 2.5 | ~40 | ~5 |
| 64 | 8.5 | 2.5 | ~44 | ~5 |
| 256 | 9.5 | 3.0 | ~48 | ~5 |

**Exponents:** single_iter N^0.03, fixed N^0.02, cenic wall N^0.02

GPU saturation knee at ~N=64 (RTX 4070 Ti SUPER). Below N=64 all lines are flat. single_iter ~3x fixed, correctly reflecting step-doubling cost (3 MuJoCo evals vs 1).

### Earlier measurement (N=1..2048, warmup=50, 200 steps, Mar 28)

Used old `manual` mode (K=5 fixed division, now replaced). Per-iteration exponents ~N^0.18 for all modes at this wider N range.

All measurements confirm CENIC adaptive machinery adds **zero measurable scaling overhead**.

## Why CENIC wall time is noisier than per-iter

CENIC wall_time = K * per_iter. K varies per step_dt (K_mean 2.6-3.3 depending on N, K_max 9-14). K variation is caused by GPU thread scheduling non-determinism in MuJoCo Warp collision (see [[2026-03-28-world-divergence-root-cause|world divergence]]). The wall_time plot shows non-monotonic behavior and wide IQR bands -- this is K variance, not scaling regression.

## Previous measurement bugs (fixed)

1. **K during free-fall (Mar 28):** Manual benchmark sampled K during free-fall (before contact onset at step ~45), getting K=1. Fix: warmup >= 50 steps.
2. **median(time/K) bias (Mar 29):** `per_iter = median(time_i / K_i)` is broken when K correlates with N (world divergence inflates K at large N, suppressing the exponent). Fix: replaced `manual` mode with `single_iter` that times one iteration directly, no K division.

## Why graph replay was worse

Graph replay (`wp.capture_launch`) dispatch time scales with total thread blocks in the captured graph. At high N, more blocks per kernel = more dispatch overhead. Direct `wp.launch()` has constant CPU overhead per launch (~0.2ms for the full iteration body).

Related: [[approaches-tried|scaling approaches tried]], [[2026-03-28-world-divergence-root-cause|world divergence]] (explains K variation).
