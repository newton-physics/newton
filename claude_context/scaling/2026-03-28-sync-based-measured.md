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

## Results (N=1 to N=2048, contact_objects scene, warmup=50, 200 steps)

**Per-iteration cost (the correct metric -- normalizes out K variation):**

| N | CENIC /iter (ms) | Fixed /iter (ms) | Manual /iter (ms) |
|---|-------------------|-------------------|--------------------|
| 1 | 7.34 | 2.67 | 6.57 |
| 64 | 9.57 | 4.10 | 8.82 |
| 256 | 11.90 | 5.44 | 11.01 |
| 1024 | 22.94 | 10.79 | 21.91 |
| 2048 | 34.65 | 15.96 | 35.47 |

**Per-iteration exponents:**
- CENIC: **N^0.174**
- Fixed: N^0.190
- Manual: N^0.184

All three modes scale identically (~N^0.18). CENIC adaptive machinery adds **zero measurable scaling overhead**.

## Why CENIC wall time is noisier than per-iter

CENIC wall_time = K * per_iter. K varies per step_dt (K_mean 2.6-3.3 depending on N, K_max 9-14). K variation is caused by GPU thread scheduling non-determinism in MuJoCo Warp collision (see [[2026-03-28-world-divergence-root-cause|world divergence]]). The wall_time plot shows non-monotonic behavior and wide IQR bands -- this is K variance, not scaling regression.

## Previous measurement bug (fixed)

The manual benchmark was sampling K during free-fall (sim_time 0.20-0.30s, before contact onset at 0.45s), getting k_fixed=1. This made manual appear 1.5-2x faster than CENIC. Fix: warmup >= 50 steps to reach contact phase before sampling K.

## Why graph replay was worse

Graph replay (`wp.capture_launch`) dispatch time scales with total thread blocks in the captured graph. At high N, more blocks per kernel = more dispatch overhead. Direct `wp.launch()` has constant CPU overhead per launch (~0.2ms for the full iteration body).

Related: [[approaches-tried|scaling approaches tried]], [[2026-03-28-world-divergence-root-cause|world divergence]] (explains K variation).
