---
aliases:
  - error metric
  - inf-norm
date: 2026-03-19
status: current
commit: c0273b9
tags:
  - solver
  - error-control
  - current
---

# Error metric evolution: RMS -> L2 -> inf-norm(q)

## Metrics tried

| Metric | Problem |
|--------|---------|
| RMS(q, qd) | Model-size-dependent (divides by 2*n_dof), velocity dominates during contact |
| L2(q, qd) | Still model-size-dependent (grows with sqrt(n_dof)), still mixing units |
| **inf-norm(q only)** | **Current.** Model-size-independent, unit-pure |

## Current: `_inf_norm_q_error_kernel`

```
max_i |q_full_i - q_double_i|
```

- tol=1e-3 means "no single joint deviates by more than 1 mm / 1 mrad"
- Same tol works for 2-DOF and 100-DOF models
- No spurious rejections during contact (velocity spikes excluded)

## Also changed in c0273b9

- `_accepted_error` array: `last_error` returns error from last *accepted* step
- `last_raw_error` property: most recent attempt (for diagnostics)
- nconmax/njmax auto-scaling removed (user responsibility)

## Key learnings

> [!important]
> - RMS and L2 are model-size-dependent -- bad for arbitrary models
> - Mixing position and velocity units requires careful scaling; simpler to just drop velocity
> - Accepted vs raw error distinction matters for user-facing displays

Related: [[2026-03-13-drake-error-controller|Drake controller]], [[2026-03-28-world-divergence-root-cause|world divergence]], [[2026-03-28-sync-based-measured|current architecture]].
