# Sync-based step_dt

**Date:** 2026-03-13
**Area:** scaling
**Status:** abandoned
**Commit:** c2ec695
**Tags:** #scaling #abandoned

## Goal
Implement the adaptive inner loop for step_dt with minimal complexity.

## Approach
Fixed CUDA graph per iteration body, replayed K times via Python loop. Used `.numpy()` on a 4-byte boundary flag to decide continue/break. `_apply_dt_cap` launched outside graph between iterations.

## Results
NOT MEASURED. Replaced with capture_while (approach 3) without benchmarking.

## Verdict
Assumed `.numpy()` per iteration was a performance problem but never verified with data. The assumption may have been wrong -- a single 4-byte device-to-host transfer per iteration is cheap compared to kernel launch overhead. Should have measured before replacing.
