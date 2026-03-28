# Architecture Overhaul: step_dt, Drake Controller, Benchmark Suite

**Date:** 2026-03-12
**Area:** solver
**Status:** success
**Commit:** cc15bed
**Tags:** #solver #success

## Goal
Mature the solver: add `step_dt()` boundary loop, switch to Drake CalcAdjustedStepSize, add contact benchmark infrastructure, clean up API.

## Approach

### Solver changes
- `SolverMuJoCoCENIC` now extends `SolverMuJoCo` directly (removed intermediate class)
- Parameters renamed: `dt_init` -> `dt_inner_init`, `dt_min` -> `dt_inner_min`, etc.
- `step_dt()` added: boundary loop using `_boundary_reset` / `_boundary_check` / `_boundary_advance` kernels
- Error controller split into `_rms_error_kernel` + `_calc_adjusted_step` (matching Drake CalcAdjustedStepSize)
- `_apply_dt_cap` separates ideal_dt from actual dt -- prevents boundary cap from corrupting controller state
- `ideal_dt` stored separately so controller can recover after contact-dense phases
- Auto-computed nconmax/njmax defaults based on shapes_per_world (later removed in Phase 5)
- qd error scaled by dt to get position units

### Infrastructure
- CLAUDE.md: 82 lines of project conventions (sim loop pattern, zero-transfer rule, dt parameter rules, viewer integration, plotting conventions)
- Viewer: `viewer.render(state, sim_time)` drives camera from simulation time, not wall clock
- Scripts reorganized: `scripts/` -> `scripts/control/`, `scripts/testing/`
- New benchmark scripts: `cenic_contact_objects.py`, `cenic_benchmark_plots.py`, `cenic_scaling_diag.py`

## Results
576 net lines added to solver. step_dt boundary loop functional. Drake CalcAdjustedStepSize controller matching reference implementation. Contact benchmark infrastructure operational with live viewer and headless modes.

## Verdict
This commit is the architectural foundation that all later work built on. The solver now has `step()` (non-graph, for debugging) and `step_dt()` (graph-optimized) paths, proper Drake controller, and full benchmark infrastructure. The ideal_dt / actual_dt separation was critical -- without it, the boundary cap at end-of-DT permanently shrinks dt and the controller can never grow it back.
