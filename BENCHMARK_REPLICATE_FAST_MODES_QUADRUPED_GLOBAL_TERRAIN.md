# Benchmark: Replicate Modes (Public Quadruped + Shared Global Terrain)

This benchmark is fully self-contained and uses:
- Bundled robot asset: `newton/examples/assets/quadruped.urdf`
- One shared procedural uneven terrain mesh (added once globally, not replicated)

## Setup

- Date: 2026-02-16
- Device: `cuda:0` (`NVIDIA GeForce RTX 4090`)
- World counts: `256`, `2048`, `4096`
- Modes: `legacy`, `auto`, `fast`
- Repeats per point: `5` (reported below as median)
- Spacing: `(0.0, 0.0, 0.0)` (homogeneous co-located worlds)
- Command:
  - `uv run python scripts/benchmarks/bench_quadruped_replicate_global_terrain.py --num-worlds 256 2048 4096 --mode all --spacing 0.0 0.0 0.0 --runs 5 --device cuda:0 --terrain-seed 42 --out-dir benchmark_quadruped_replicate_global_terrain_20260216_231128_effects`
- Artifact root:
  - `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects`

## Isolation Scope

- This document isolates **replicate mode effects** (`legacy` vs `auto` vs `fast`) on the same code revision.
- Incremental commit-by-commit isolation is documented separately in:
  - `BENCHMARK_REPLICATE_FAST_MODES_QUADRUPED.md`

## Metrics

Timed phases per run:
- `robot_build`: quadruped template build + `add_urdf(...)`
- `replicate`: `scene.replicate(..., mode=...)`
- `terrain_build_once`: global terrain mesh generation + insertion once
- `finalize`: `scene.finalize()`
- `replicate_path_only = robot_build + replicate + finalize`
- `total_startup = robot_build + replicate + terrain_build_once + finalize`

## Results (Median Seconds)

### `num_worlds = 256`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | speedup vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0060 | 0.0075 | 0.0096 | 0.0416 | 0.0555 | 0.0652 | +0.0% | 1.000x |
| `auto` | 0.0060 | 0.0066 | 0.0096 | 0.0317 | 0.0440 | 0.0536 | -17.8% | 1.216x |
| `fast` | 0.0060 | 0.0065 | 0.0097 | 0.0315 | 0.0441 | 0.0539 | -17.3% | 1.209x |

### `num_worlds = 2048`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | speedup vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0062 | 0.0764 | 0.0101 | 0.5797 | 0.7016 | 0.7117 | +0.0% | 1.000x |
| `auto` | 0.0064 | 0.1154 | 0.0105 | 0.4503 | 0.5737 | 0.5838 | -18.0% | 1.219x |
| `fast` | 0.0063 | 0.0630 | 0.0105 | 0.4593 | 0.5813 | 0.5918 | -16.8% | 1.203x |

### `num_worlds = 4096`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | speedup vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0062 | 0.2414 | 0.0099 | 1.2701 | 1.5172 | 1.5271 | +0.0% | 1.000x |
| `auto` | 0.0062 | 0.1892 | 0.0098 | 0.9748 | 1.1594 | 1.1903 | -22.1% | 1.283x |
| `fast` | 0.0062 | 0.1866 | 0.0100 | 1.0181 | 1.1987 | 1.2140 | -20.5% | 1.258x |

## Effect Decomposition (Vs `legacy`)

This section reports the **effect of mode changes by phase**, not only overall totals. Values come from `summary_effects_vs_legacy.csv`.

| worlds | mode | replicate Δ% | finalize Δ% | total Δ% | replicate contribution to total Δ | finalize contribution to total Δ |
|---|---|---:|---:|---:|---:|---:|
| `256` | `auto` | -13.0% | -23.9% | -17.8% | 8.5% | 86.1% |
| `256` | `fast` | -14.2% | -24.2% | -17.3% | 9.5% | 89.6% |
| `2048` | `auto` | +50.9% | -22.3% | -18.0% | -30.4% | 101.2% |
| `2048` | `fast` | -17.6% | -20.8% | -16.8% | 11.2% | 100.4% |
| `4096` | `auto` | -21.6% | -23.2% | -22.1% | 15.5% | 87.7% |
| `4096` | `fast` | -22.7% | -19.8% | -20.5% | 17.5% | 80.5% |

Notes:
- Positive contribution means that phase helped reduce startup time.
- Negative contribution means that phase moved in the opposite direction (for example `2048/auto` replicate phase regressed while finalize improved enough to dominate net speedup).

## Reference: Internal Excavation (M445, Not Public-Self-Contained)

This section is reference-only and intentionally excluded from self-contained test artifacts.

- Current run (same machine, current branch):
  - Command:
    - `uv run python scripts/benchmark/benchmark_excavation_w_cabin_analytic_throughput.py --num-envs 2048 --warmup-steps 0 --benchmark-steps 5 --output-dir outputs/benchmark_w_cabin_analytic_throughput/pr_ref_20260216_224801_newton_replicate_modes`
  - Result:
    - `env_build_seconds = 10.666`
    - `env_steps_per_second = 25014.71`

- Comparison baseline/best from `BENCHMARK_REPLICATE_FAST_MODES_M445.md`:
  - Baseline (`71f669f`): `env_build_seconds = 22.048`, `env_steps_per_second = 9021.0`
  - Best prior build (`defa65a`): `env_build_seconds = 9.848`
  - Best prior throughput in that sweep (`1491e83`): `env_steps_per_second = 20198.0`

- Speedups:
  - Build vs baseline: `-51.6%` (faster)
  - Throughput vs baseline: `+177.3%`
  - Build vs best prior build: `+8.3%` (slower)
  - Throughput vs best prior throughput: `+23.8%`

## Raw Artifacts

- Quadruped summary CSV: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects/summary.csv`
- Quadruped summary JSON: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects/summary.json`
- Quadruped effects CSV: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects/summary_effects_vs_legacy.csv`
- Quadruped effects JSON: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects/summary_effects_vs_legacy.json`
- Quadruped per-run records: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_231128_effects/run_records.json`
- Excavation reference run: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_ref_20260216_224801_newton_replicate_modes/benchmark_results.json`
