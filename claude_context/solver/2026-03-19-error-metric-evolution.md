# Error Metric Evolution: RMS -> L2 -> Inf-Norm(q)

**Date:** 2026-03-19 to 2026-03-20
**Area:** solver
**Status:** success
**Commit:** b1d2148 (L2), c0273b9 (inf-norm)
**Tags:** #solver #success #current

## Goal
Find the right error metric for step-size control that is physically meaningful, model-size-independent, and robust during contact.

## Approach

### Metric 1: RMS over joint_q + joint_qd (original, 3380670)
- `_rms_error_kernel`: sqrt(sum((q_full - q_double)^2 + (qd_full * dt - qd_double * dt)^2) / (2*n_dof))
- qd scaled by dt to convert to position-like displacement units
- Problem: RMS divides by 2*n_dof, making tol model-size-dependent (a 100-DOF model needs much tighter tol than a 2-DOF model to get the same per-joint accuracy). Also, velocity terms dominated during contact events.

### Metric 2: L2 norm on joint_q + joint_qd (b1d2148)
- Dropped the /n normalization (just straight L2 norm)
- Problem: still model-size-dependent (L2 grows with sqrt(n_dof)), still mixing position and velocity units

### Metric 3: Inf-norm on joint_q only (c0273b9) -- CURRENT
- `_inf_norm_q_error_kernel`: max_i |q_full_i - q_double_i|
- Dropped joint_qd entirely
- Inf-norm is model-size-independent: tol means "no single joint can have more than tol error"
- Units are pure position (rad for revolute, m for prismatic) -- tol is directly interpretable

### Additional changes in c0273b9
- `_accepted_error` array added: `last_error` now returns error from last *accepted* step, not last attempt
- `last_raw_error` property added for most recent attempt error (for diagnostics)
- nconmax/njmax auto-scaling removed from solver constructor (moved to user responsibility)
- Benchmark plots completely rewritten to use subprocess isolation and JSON result caching

## Results
Inf-norm on joint_q provides stable, interpretable error control:
- tol=1e-3 means "no joint deviates by more than 1mm/1mrad between full and half steps"
- Model-size-independent: same tol works for 2-DOF and 100-DOF models
- No spurious rejections during contact (velocity spikes no longer pollute the error signal)

## Verdict
Inf-norm on joint_q only is the correct metric. Key learnings:
1. RMS and L2 are model-size-dependent -- bad for a solver that runs arbitrary models
2. Mixing position and velocity units requires careful scaling; easier and more robust to just drop velocity
3. The accepted vs. raw error distinction is important for user-facing status displays (showing rejected-step error is misleading)
4. nconmax auto-scaling was removed because the heuristic was fragile and model-dependent; better for users to set explicitly
