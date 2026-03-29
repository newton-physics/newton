# Benchmark single_iter mode and plot cleanup

**Date:** 2026-03-29
**Status:** approved

## Problem

The current `manual` benchmark mode runs K=5 calls to `_run_iteration_body()` and divides by K to get per-iteration cost. This produces a per_iter exponent (0.161) that is nearly identical to CENIC's (0.149), and CENIC's exponent is lower than fixed-step's (0.169). CENIC doing 3x the work per iteration cannot scale better than a single eval. The `median(time_i / K_i)` metric is broken because K correlates with N (world divergence causes K to inflate at large N, suppressing the exponent).

## Solution

Replace `manual` with `single_iter`: time a single `_run_iteration_body()` call directly, sync-to-sync, from a saved contact-phase state. No K division.

## Modes

| Mode | Measures | Implementation |
|------|----------|----------------|
| `cenic` (was `graph`) | Total throughput per `step_dt()` | Full adaptive step, variable K |
| `fixed` | Baseline throughput per `step()` | Single MuJoCo eval at dt=10ms |
| `single_iter` (was `manual`) | Raw per-iteration GPU cost | One `_run_iteration_body()`, sync before/after |

## single_iter implementation

1. Build model, create CENIC solver
2. Warmup with `step_dt` for `max(warmup, 50)` steps to reach contact phase
3. Save contact-phase state: solver internal buffers (joint_q, joint_qd, body_q, body_qd, dt, sim_time)
4. For each of `steps` timed samples:
   - Restore saved state into solver internals
   - `wp.synchronize()`
   - `t0 = time.perf_counter()`
   - `solver._run_iteration_body(effective_dt_max)`
   - `wp.synchronize()`
   - Record `time.perf_counter() - t0`
5. K is always 1. `per_iter_median = median(times)`.

## Plots (5 total)

1. **Wall time vs N** -- cenic + fixed, log-log, exponents, IQR bands
2. **Per-iteration cost vs N** -- single_iter + fixed, log-log, exponents. Expect matching exponents, ~3x ratio.
3. **K vs N** -- cenic only, K_mean + K_max + IQR
4. **Amortization** -- cost per world from wall time, cenic + fixed
5. **Error trace** -- N=1 cenic, error vs sim time with tol line (unchanged)

## File changes

- `scripts/bench/benchmarks/_manual_step.py` -- rewrite as `_single_iter.py`
- `scripts/bench/benchmarks/scaling.py` -- rename `graph` to `cenic`, replace `manual` with `single_iter`, update plot functions to match the 5-plot layout
- `scripts/bench/plotting.py` -- update STYLES dict (rename `graph` to `cenic`, replace `manual` with `single_iter`)
- `scripts/bench/infra.py` -- no changes needed

## What this proves

- Plot 2: single_iter and fixed have matching exponents (zero CENIC scaling overhead)
- Plot 2: single_iter is ~3x fixed (step-doubling cost: 3 evals vs 1)
- Plot 1: real-world throughput with natural K variation
- Plot 5: adaptive stepping mechanism is working
