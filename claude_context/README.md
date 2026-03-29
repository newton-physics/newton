---
tags:
  - index
---
## CENIC Implementation Log
> [!tip] Read before proposing new work
> Check existing notes before retrying anything. Every approach gets a note, especially failures.

## Current architecture

- [[2026-03-28-sync-based-measured|Direct-launch step_dt]] -- wp.launch() + Python boundary loop. N^0.174 per-iter. **Active implementation.**

## Scaling

- [[approaches-tried|Approaches tried]] -- graph replay, capture_while, direct launch, benchmark bug. All attempts in one place.
- [[2026-03-28-world-divergence-root-cause|World divergence]] -- global atomics in MuJoCo Warp collision cause K variation. **Upstream issue.**

## Solver

- [[2026-03-12-architecture-overhaul|Architecture overhaul]] -- step_dt, Drake controller, ideal_dt/actual_dt separation.
- [[2026-03-13-drake-error-controller|Drake error controller]] -- CalcAdjustedStepSize, hysteresis bug, accept-at-floor.
- [[2026-03-19-error-metric-evolution|Error metric]] -- RMS -> L2 -> inf-norm(q only).

## Benchmarking

- [[2026-03-12-benchmark-infrastructure|Benchmark infrastructure]] -- scenes, bench platform, fresh solver per measurement.
- Benchmark modes (Mar 29): `cenic` (wall time), `fixed` (baseline), `single_iter` (raw per-iter GPU cost). `manual` and `capture_while_isolation` removed.

## Roadmap

- [[next-steps|Next steps]] -- what's blocking better scaling, possible directions.
