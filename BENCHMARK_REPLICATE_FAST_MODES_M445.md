# Benchmark: Replicate Fast Modes (Internal M445 Analytic)

This benchmark uses the internal M445 analytic environment/URDF for performance measurement only.

## Setup

- Date: 2026-02-16
- Device: `cuda:0`
- Command:
  - `uv run python scripts/benchmark/benchmark_excavation_w_cabin_analytic_throughput.py --num-envs 2048 --warmup-steps 0 --benchmark-steps 5 --output-dir <per-commit-dir>`
- Results root:
  - `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018`

## Results (one run per commit)

| Commit | Change | env_build_seconds | Δ vs baseline | Δ vs previous | env_steps_per_second |
|---|---|---:|---:|---:|---:|
| `71f669f` | Baseline (pre-optimization) | 22.048s | +0.0% | +0.0% | 9021.0 |
| `defa65a` | replicate_physics fast mode + CUDA explicit contact pair precompute | 9.848s | -55.3% | -55.3% | 15768.6 |
| `2a4e0dd` | translation-only fast path in add_builder transform application | 10.605s | -51.9% | +7.7% | 19460.6 |
| `7ad91e7` | skip zero-offset transforms + skip custom-attribute merge setup when empty | 12.252s | -44.4% | +15.5% | 19917.1 |
| `1491e83` | avoid global filter-pair dedup/sort for replica-only storage | 11.040s | -49.9% | -9.9% | 20198.0 |
| `a596962` | dedicated zero-spacing replicate branch (no offset grid) | 11.417s | -48.2% | +3.4% | 13561.6 |
| `5a0a1f2` | avoid extra device copy for CUDA contact-pair slices | 11.542s | -47.7% | +1.1% | 13436.3 |

## Summary

- Best observed build time: `9.848s` at commit `defa65a`.
- Baseline build time: `22.048s`.
- Net improvement at best point: `55.3%`.
- Throughput (`env_steps_per_second`) is noisier run-to-run than build time; compare it with caution unless averaged across multiple runs.

## Raw Artifacts

- `71f669f`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/71f669f/benchmark_results.json`
- `defa65a`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/defa65a/benchmark_results.json`
- `2a4e0dd`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/2a4e0dd/benchmark_results.json`
- `7ad91e7`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/7ad91e7/benchmark_results.json`
- `1491e83`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/1491e83/benchmark_results.json`
- `a596962`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/a596962/benchmark_results.json`
- `5a0a1f2`: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_commit_bench_20260216_201018/5a0a1f2/benchmark_results.json`
