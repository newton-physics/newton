# Allocation-Free step() via StepWorkspace

**Date:** 2026-03-20
**Area:** solver
**Status:** success
**Commit:** patched in .venv (not committed to newton repo)
**Tags:** #solver #success

## Goal
Eliminate all per-step GPU allocations (wp.empty/zeros/full) from mujoco_warp's step() function to enable wp.capture_while() support. CUDA graph capture cannot tolerate allocations in the graph body.

## Approach
- Created `StepWorkspace` dataclass in `types.py` holding all pre-allocated temporary arrays
- `workspace.py` creates them once via `create_step_workspace(m, d)`
- Lazy init on first `step()` call (`d._workspace is None` check)
- All temporaries that were previously allocated per-step now pulled from workspace

### Files modified (in .venv/lib/python3.12/site-packages/mujoco_warp/_src/)
- `types.py` -- StepWorkspace dataclass + `_workspace` field on Data
- `workspace.py` -- NEW: `create_step_workspace()`
- `io.py` -- skip `_workspace` in `make_data`/`put_data` field iteration
- `forward.py` -- workspace init hook in `step()`, `implicit()`, `fwd_actuation()`
- `collision_driver.py` -- reuse collision context arrays
- `collision_convex.py` -- reuse EPA + MultiCCD arrays
- `solver.py` -- reuse SolverContext + step_size_cost + nsolving, zero grad/Mgrad/h
- `derivative.py` -- reuse deriv_vel
- `smooth.py` -- reuse ten_Jdot, ten_bias_coef, ncon, wrap_geom_xpos, ne_connect, ne_weld, subtree_bodyvel
- `passive.py` -- reuse fluid_applied

### Not patched
- `sensor.py` allocations: conditional on sensor config, CENIC models have no sensors so these never fire

## Results
- Basic CUDA graph capture works with patched step()
- wp.capture_while() works with patched step()
- CENIC contact test passes
- No regression in simulation accuracy

## Verdict
This is a prerequisite for the capture_while zero-sync architecture. The patch lives in `.venv` (local mujoco_warp install), not in the newton repo. If mujoco_warp upstream adds allocation-free step(), this patch becomes unnecessary. Until then, it must be reapplied after any mujoco_warp upgrade.
