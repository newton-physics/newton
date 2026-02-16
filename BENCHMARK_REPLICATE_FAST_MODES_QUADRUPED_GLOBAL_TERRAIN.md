# Benchmark: Replicate Modes (Public Quadruped + Shared Global Terrain)

This benchmark is fully self-contained and uses:
- Bundled robot asset: `newton/examples/assets/quadruped.urdf`
- One shared procedural uneven terrain mesh (added once globally, not replicated)

## Setup

- Date: 2026-02-16
- Device: `cuda:0` (`NVIDIA GeForce RTX 4090`)
- World counts: `256`, `2048`
- Modes: `legacy`, `auto`, `fast`
- Repeats per point: `5` (reported below as median)
- Spacing: `(0.0, 0.0, 0.0)` (homogeneous co-located worlds)
- Command:
  - `uv run python scripts/benchmarks/bench_quadruped_replicate_global_terrain.py --num-worlds 256 2048 --mode all --spacing 0.0 0.0 0.0 --runs 5 --device cuda:0 --terrain-seed 42 --out-dir benchmark_quadruped_replicate_global_terrain_20260216_205459`
- Artifact root:
  - `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_205459`

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
| `legacy` | 0.0062 | 0.0078 | 0.0097 | 0.0453 | 0.0600 | 0.0725 | +0.0% | +0.0% |
| `auto` | 0.0061 | 0.0065 | 0.0097 | 0.0325 | 0.0450 | 0.0547 | -24.5% | -25.0% |
| `fast` | 0.0061 | 0.0064 | 0.0096 | 0.0313 | 0.0438 | 0.0537 | -26.0% | -27.0% |

### `num_worlds = 2048`

| mode | robot_build | replicate | terrain_build_once | finalize | replicate_path_only | total_startup | Δ total vs legacy | Δ replicate_path_only vs legacy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 0.0063 | 0.1256 | 0.0102 | 0.6037 | 0.7185 | 0.7287 | +0.0% | +0.0% |
| `auto` | 0.0063 | 0.1119 | 0.0101 | 0.4802 | 0.5798 | 0.5897 | -19.1% | -19.3% |
| `fast` | 0.0063 | 0.1143 | 0.0101 | 0.4706 | 0.5913 | 0.6030 | -17.2% | -17.7% |

## Interpretation

- Shared global terrain adds a mostly constant startup cost (~10 ms median) and does not dominate build time.
- At `2048` worlds, the startup bottleneck remains in replication/finalization, where `auto/fast` reduce median startup by ~17-19% vs `legacy`.
- At `256` worlds, gains are ~25-27% on median startup.

## Why M445 Is Slower Than Quadruped

Compared with this bundled quadruped, M445 startup is much slower because:
- The robot model and scene bookkeeping are larger (more links/joints/shapes and heavier metadata propagation).
- Collision/filter structures grow with a larger per-world footprint, making replication/finalize costlier.
- Task-specific setup in M445 analytic environments adds extra CPU-side construction work beyond bare robot replication.

In short: replicate mode optimizations help both cases, but M445 has a larger constant and per-world construction burden.

## Raw Artifacts

- Summary CSV: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_205459/summary.csv`
- Summary JSON: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_205459/summary.json`
- Per-run records: `/home/lorenzo/moleworks/newton/benchmark_quadruped_replicate_global_terrain_20260216_205459/run_records.json`
