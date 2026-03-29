---
aliases:
  - next steps
  - roadmap
date: 2026-03-28
status: active
tags:
  - roadmap
---

# Next steps

## What we have

- Adaptive solver (CENIC): step doubling, Drake error controller, inf-norm on q, accept-at-floor
- N^0.174 per-iteration scaling -- matches base MuJoCo Warp kernel cost, zero CENIC overhead
- Benchmark suite with correct K sampling during contact phase

## Known blockers (upstream)

- **Graph replay dispatch scales with thread blocks** -- CUDA runtime issue, not fixable from Python. Why direct `wp.launch()` beats graph replay. See [[approaches-tried]].
- **Shared contact counter causes world divergence** -- all N worlds share one atomic counter for contact slots. Causes K variation and noisy wall-time. Fix requires per-world counters upstream in MuJoCo Warp. See [[2026-03-28-world-divergence-root-cause]].
- **N^0.18 floor is GPU physics cost** -- sub-linear (good), but it's the floor for any dispatch strategy.

## Possible directions

- **API alignment**: CENIC as drop-in replacement for SolverMuJoCo (same step() signature, state types, control interface)
- **Larger scenes**: ANYmal quadruped, humanoid -- more DOFs, more contacts, stiffer dynamics
- **Paper benchmarks**: accuracy vs wall-time tradeoffs, CENIC at various tol vs fixed at various dt, statistical treatment of K variation
- **Upstream contributions**: per-world contact counter issue/PR, graph replay dispatch scaling issue
- **Tolerance auto-tuning**: adaptive tol based on physical regime (tighter during contact, looser in free-flight)
