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
  - `uv run python scripts/benchmarks/bench_quadruped_replicate_global_terrain.py --num-worlds 256 2048 4096 --mode all --spacing 0.0 0.0 0.0 --runs 5 --device cuda:0 --terrain-seed 42 --out-dir benchmark_quadruped_replicate_global_terrain_20260216_224715_w256_2048_4096`
- Artifact root:
  - `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_224715_w256_2048_4096`

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

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | Δ replicate_path_only vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0060 | 0.0076 | 0.0096 | 0.0413 | 0.0550 | 0.0646 | +0.0% | +0.0% |
| `auto` | 0.0060 | 0.0067 | 0.0095 | 0.0318 | 0.0447 | 0.0543 | -15.9% | -18.6% |
| `fast` | 0.0061 | 0.0065 | 0.0095 | 0.0314 | 0.0439 | 0.0534 | -17.3% | -20.1% |

### `num_worlds = 2048`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | Δ replicate_path_only vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0063 | 0.1282 | 0.0099 | 0.5921 | 0.7126 | 0.7227 | +0.0% | +0.0% |
| `auto` | 0.0063 | 0.1186 | 0.0101 | 0.4544 | 0.5794 | 0.5895 | -18.4% | -18.7% |
| `fast` | 0.0062 | 0.1147 | 0.0104 | 0.4570 | 0.5750 | 0.5969 | -17.4% | -19.3% |

### `num_worlds = 4096`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | Δ replicate_path_only vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0063 | 0.2339 | 0.0098 | 1.3136 | 1.5720 | 1.5818 | +0.0% | +0.0% |
| `auto` | 0.0063 | 0.2008 | 0.0098 | 1.1127 | 1.2771 | 1.2868 | -18.6% | -18.8% |
| `fast` | 0.0063 | 0.2078 | 0.0102 | 1.0420 | 1.2574 | 1.2676 | -19.9% | -20.0% |

## Interpretation

- Shared global terrain adds a mostly constant startup cost (~10 ms median) and does not dominate build time.
- At `2048` and `4096` worlds, startup bottlenecks remain in replication/finalization.
- For the RL-like `4096` case, `auto/fast` reduce median startup by roughly `19-20%` vs `legacy`.

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

- Quadruped summary CSV: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_224715_w256_2048_4096/summary.csv`
- Quadruped summary JSON: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_224715_w256_2048_4096/summary.json`
- Quadruped per-run records: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_224715_w256_2048_4096/run_records.json`
- Excavation reference run: `/home/lorenzo/moleworks/moleworks_newton/outputs/benchmark_w_cabin_analytic_throughput/pr_ref_20260216_224801_newton_replicate_modes/benchmark_results.json`
