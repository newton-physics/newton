# Initial CENIC Solver Creation

**Date:** 2026-03-01
**Area:** solver
**Status:** success (partial)
**Commit:** 3380670
**Tags:** #solver #success

## Goal
Create the CENIC adaptive-step solver from scratch -- step doubling with per-world dt adaptation, entirely on GPU.

## Approach
- `SolverMuJoCoCENIC` extending `SolverVariableStepMuJoCo` (a thin intermediate class, later removed in Phase 3)
- 3-eval step doubling: full dt, dt/2, dt/2
- RMS error between scratch_full and scratch_double across both joint_q and joint_qd
- Custom error controller with NaN guard, accept-at-floor, growth hysteresis (1.2x), shrink hysteresis
- Per-world `_select_*_kernel` kernels for conditional state copy (accept/reject per world)
- All decision logic in Warp kernels, no `.numpy()` in hot path

### Files created
- `newton/_src/solvers/mujoco/solver_mujoco_cenic.py` (498 lines)
- `newton/_src/solvers/mujoco/solver_variable_step_mujoco.py` (192 lines, later deleted)
- Exports in `__init__.py` and `newton/solvers.py`
- Demo scripts: `cenic_step_anymal_walk.py`, `cenic_step_quadruped.py`, `variable_step_double_pendulum.py`

## Results
Solver works for basic scenes (double pendulum, quadruped). Step doubling produces correct error estimates. Per-world adaptive dt functional.

## Verdict
Functional foundation but incomplete:
- No `step_dt()` boundary loop -- only `step()` doing one adaptive attempt per call
- No CUDA graph capture
- No viewer sim-time integration
- Error controller was hand-rolled (not yet matching Drake CalcAdjustedStepSize)
- The `SolverVariableStepMuJoCo` intermediate class added no value and was removed in the March 12 overhaul
- RMS error mixing position and velocity units was later identified as problematic (Phase 5)
