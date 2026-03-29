---
aliases:
  - architecture overhaul
date: 2026-03-12
status: success
commit: cc15bed
tags:
  - solver
  - architecture
  - success
---

# Architecture overhaul: step_dt, Drake controller, benchmark suite

> [!info] Foundation commit
> This is the architectural foundation all later work built on.

## What changed

- `SolverMuJoCoCENIC` extends `SolverMuJoCo` directly (removed intermediate class)
- `step_dt()` added: boundary loop via `_boundary_reset` / `_boundary_check` / `_boundary_advance` kernels
- [[2026-03-13-drake-error-controller|Drake CalcAdjustedStepSize]] controller
- `_apply_dt_cap` separates `ideal_dt` from actual `dt` -- prevents boundary cap from corrupting controller state
- [[2026-03-12-benchmark-infrastructure|Benchmark infrastructure]]: contact scenes, scaling plots, viewer sim-time integration

## Critical design decision

> [!important] ideal_dt / actual_dt separation
> The boundary cap at end-of-DT sets `dt = min(ideal_dt, remaining_time)` but does NOT write back to `ideal_dt`. Without this, the controller permanently shrinks dt and can never grow it back.

## Results

576 net lines. `step_dt()` is the primary stepping path (originally graph-optimized, now direct-launch). Full [[2026-03-12-benchmark-infrastructure|benchmark infrastructure]].

Related: [[2026-03-13-drake-error-controller|Drake controller]], [[2026-03-28-sync-based-measured|current architecture]].
