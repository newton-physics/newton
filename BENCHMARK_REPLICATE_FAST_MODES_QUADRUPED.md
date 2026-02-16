# Benchmark: Replicate Fast Modes (Public Quadruped URDF)

This benchmark is fully self-contained and uses the bundled URDF asset: `newton/examples/assets/quadruped.urdf`.

## Setup

- Date: 2026-02-16
- Device: `cuda:0`
- Worlds: `2048`
- Repeats per commit: `5` (reported metric is median)
- Timed phases per repeat:
  - `parse`: `ModelBuilder.add_urdf(...)`
  - `replicate`: `scene.replicate(template, ...)`
  - `finalize`: `scene.finalize(...)`
- Results root:
  - `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550`

## Results (median over 5 runs)

| Commit | Change | parse | replicate | finalize | total | Δ vs baseline | Δ vs previous |
|---|---|---:|---:|---:|---:|---:|---:|
| `71f669f` | Baseline (pre-optimization) | 0.007s | 0.175s | 0.363s | 0.600s | +0.0% | +0.0% |
| `defa65a` | replicate_physics fast mode + CUDA explicit contact pair precompute | 0.013s | 0.372s | 0.751s | 1.131s | +88.6% | +88.6% |
| `2a4e0dd` | translation-only fast path in add_builder transform application | 0.007s | 0.570s | 0.391s | 0.968s | +61.4% | -14.4% |
| `7ad91e7` | skip zero-offset transforms + skip custom-attribute merge setup when empty | 0.007s | 0.061s | 0.390s | 0.469s | -21.9% | -51.6% |
| `1491e83` | avoid global filter-pair dedup/sort for replica-only storage | 0.007s | 0.063s | 0.386s | 0.453s | -24.5% | -3.3% |
| `a596962` | dedicated zero-spacing replicate branch (no offset grid) | 0.007s | 0.059s | 0.379s | 0.446s | -25.6% | -1.5% |
| `5a0a1f2` | avoid extra device copy for CUDA contact-pair slices | 0.007s | 0.060s | 0.378s | 0.454s | -24.3% | +1.8% |

## Summary

- Baseline total median: `0.600s` at `71f669f`.
- Best total median: `0.446s` at `a596962`.
- Net improvement at best point: `25.6%`.
- Main gain source is replication path simplification for zero-spacing homogeneous worlds; parse time is mostly unchanged across commits.

## Raw Artifacts

- `71f669f`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/71f669f.json`
- `defa65a`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/defa65a.json`
- `2a4e0dd`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/2a4e0dd.json`
- `7ad91e7`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/7ad91e7.json`
- `1491e83`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/1491e83.json`
- `a596962`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/a596962.json`
- `5a0a1f2`: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_20260216_201550/5a0a1f2.json`
