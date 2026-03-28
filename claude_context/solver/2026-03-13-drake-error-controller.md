# Drake CalcAdjustedStepSize Error Controller

**Date:** 2026-03-13
**Area:** solver
**Status:** success
**Commit:** cc15bed (initial), c0273b9 (inf-norm switch)
**Tags:** #solver #success #current

## Goal
Replace the hand-rolled error controller with Drake's CalcAdjustedStepSize for well-characterized step-size adaptation.

## Approach
Implemented as `_calc_adjusted_step` Warp kernel. Parameters:
- kSafety = 0.9 (conservative factor on ideal step size)
- kMinShrink = 0.1 (never shrink more than 10x)
- kMaxGrow = 5.0 (never grow more than 5x)
- kHysteresisHigh = 1.2 (only reject if error > 1.2 * tol)
- err_order = 2 (step doubling is second-order)

### Key design decisions
- `ideal_dt` stored separately from actual `dt` -- the boundary cap at end-of-DT sets actual dt = min(ideal_dt, remaining_time), but does not write back to ideal_dt. This prevents the controller from losing its growth trajectory.
- `_apply_dt_cap` kernel enforces dt_min/dt_max and the boundary cap without corrupting controller state.
- Accept-at-floor: if dt == dt_min, always accept regardless of error (prevents infinite rejection loops).
- NaN guard: if error is NaN, shrink to dt_min immediately.

### Bug fix history
1. **Hysteresis guard suppressing rejections** (fixed Mar 13): The Drake hysteresis check was `if error < kHysteresisHigh * tol: accept` but this was applied BEFORE the reject path, meaning it was suppressing ALL rejections (not just tiny grows). Every step was accepted regardless of error. Fixed by restructuring: compute new dt first, then apply hysteresis only to decide accept/reject.
2. **Velocity units in error** (fixed Mar 19-20): RMS error mixed position (rad/m) and velocity (rad/s, m/s) units, causing velocity-dominated rejection during contact. Switched to inf-norm on joint_q only (c0273b9).

## Results
Controller correctly tracks error at tol boundary. With tol=1e-3 and MuJoCo implicit Euler, error is O(dt^2) ~ 4e-6 at dt=0.002, so dt stays at dt_max (correct behavior -- error is well below tolerance). To observe dt variation, use tighter tol (1e-6+) or larger DT_OUTER (0.01+).

## Verdict
Drake CalcAdjustedStepSize is the correct choice. The hysteresis bug was subtle -- the solver appeared to work but was never actually rejecting steps. The ideal_dt / actual_dt separation is load-bearing and must be preserved in any future refactors.
