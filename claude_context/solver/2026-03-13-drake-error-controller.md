---
aliases:
  - Drake controller
  - CalcAdjustedStepSize
date: 2026-03-13
status: current
commit: cc15bed
tags:
  - solver
  - error-control
  - current
---

# Drake CalcAdjustedStepSize error controller

Implemented as `_calc_adjusted_step` Warp kernel. Replaces hand-rolled controller.

## Parameters

| Constant | Value | Meaning |
|----------|-------|---------|
| kSafety | 0.9 | Conservative factor on ideal step size |
| kMinShrink | 0.1 | Never shrink more than 10x |
| kMaxGrow | 5.0 | Never grow more than 5x |
| kHysteresisHigh | 1.2 | Suppress tiny grows |
| err_order | 2 | Step doubling is second-order |

## Design decisions

- `ideal_dt` stored separately from `dt` (see [[2026-03-12-architecture-overhaul|architecture overhaul]])
- Accept-at-floor: if `dt == dt_min`, always accept (prevents infinite rejection loops)
- NaN guard: shrink to `dt_min` immediately

## Bugs found

> [!bug] Hysteresis suppressing ALL rejections (fixed Mar 13)
> Hysteresis check applied BEFORE reject path, meaning every step was accepted regardless of error. Solver appeared to work but never rejected anything.

> [!bug] Velocity units in error (fixed Mar 19-20)
> RMS error mixed position and velocity units, causing velocity-dominated rejection during contact. Fixed by switching to [[2026-03-19-error-metric-evolution|inf-norm on joint_q only]].

> [!info] CUDA graph capture removed (Mar 28)
> Graph capture/replay was removed from the solver. All kernels now dispatched via direct `wp.launch()`. See [[approaches-tried|scaling approaches tried]].

## Behavior

With tol=1e-3 and MuJoCo implicit Euler, error is O(dt^2) ~ 4e-6 at dt=0.002. dt stays at dt_max (correct -- error well below tolerance). To see dt variation: use tighter tol (1e-6+) or larger DT_OUTER (0.01+).

Related: [[2026-03-19-error-metric-evolution|error metric evolution]], [[2026-03-12-architecture-overhaul|architecture overhaul]].
