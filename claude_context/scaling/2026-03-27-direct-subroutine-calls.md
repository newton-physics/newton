# Direct subroutine calls + graph_conditional=False

**Date:** 2026-03-27
**Area:** scaling
**Status:** failed
**Commit:** reverted (was on branch marco/cenic-upstream-integration, deleted)
**Tags:** #scaling #failed

## Goal
Reduce graph node count by bypassing mujoco_warp.step() and calling sub-functions directly. Disable solver.solve() conditional graph to eliminate inner capture_while overhead.

## Approach
Called fwd_position/fwd_velocity/fwd_actuation/fwd_acceleration/solver.solve/euler directly instead of mujoco_warp.step(). Set graph_conditional=False so solver.solve() unrolls as fixed kernel launches (no inner conditional graph).

## Results
- **Graph:** N^0.126
- **Manual:** N^0.003
- N=1: 19.91ms, N=256: 42.69ms

## Verdict
WORSE on every metric. graph_conditional=False forces all ~100 solver iterations even after convergence (normally 3-9), making it 4x slower in absolute terms. More graph nodes from unrolled iterations made capture_while scaling worse (N^0.126 vs N^0.099). The one useful finding: manual path confirmed kernel floor is N^0.003, proving all scaling comes from CUDA graph/driver overhead, not from physics kernels.
