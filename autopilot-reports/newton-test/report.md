# Newton Physics Engine - Installation & Validation Report

**Date:** 2026-03-24
**Node:** Linux x86_64, NVIDIA L40 GPU (47 GiB, sm_89), CUDA 12.8
**Branch:** autopilot/newton-test

## Install Summary

| Item | Value |
|------|-------|
| Newton version | 1.1.0.dev0 |
| Install path | /home/horde/newton-src (editable, `pip install -e .`) |
| Source repo | https://github.com/newton-physics/newton |
| Warp version | 1.12.0 |
| CUDA Toolkit | 12.9 (driver 12.8) |
| Python | 3.10 |

### Install Notes

- The repo listed in the project spec (`NVIDIA-Omniverse/newton`) does not exist; the correct repo is `newton-physics/newton`.
- The PyPI package `newton-physics` is a redirect stub; the actual package is `newton`.
- Install succeeded with `pip install -e .` from source (installs `warp-lang>=1.12.0` automatically).
- Optional dependencies installed manually for broader test coverage:
  - `mujoco-warp` (MuJoCo backend)
  - `scipy`, `trimesh`, `GitPython`
  - `pxr` / USD not available — tests/examples requiring it were skipped.

## Newton Import Test

```
Newton version: 1.1.0.dev0
Warp 1.12.0 initialized:
  CUDA Toolkit 12.9, Driver 12.8
  Devices:
    "cpu"    : "x86_64"
    "cuda:0" : "NVIDIA L40" (47 GiB, sm_89, mempool enabled)
```

**Status: PASS**

## Rigid Body Simulation

Custom benchmark using `SolverXPBD` + `ModelBuilder` API:

### Falling Box Test (1 box, z-up gravity=-9.81)
- Initial position: (0.0, 0.0, 5.0)
- After 300 steps at dt=1/120s (2.5 simulated seconds):
  - Final position: (0.0000, 0.0000, 0.4999) ← settled at half-height, as expected
  - Wall time: 0.83s → **361 steps/sec**
- **Status: PASS** — box fell, collided with ground, and settled at correct height (z≈0.5)

### Collision Detection Test (10 boxes, stacked)
- 10 rigid boxes with ground + inter-body collision
- After 300 steps: all bodies remain above floor (min z=0.400)
- **Status: PASS** — collision detection functional

## Test Suite Results

Full suite run: `python3 -m newton.tests`

| Category | Count |
|----------|-------|
| Total tests | 2873 |
| Passed | ~1885 |
| Failed | ~23 |
| Errors | ~575 |
| Skipped | ~351 |

**Note:** The full suite run aborted with `BrokenProcessPool` (OOM or process killed in the parallel runner). The counts above are from the partial run.

### Broad Phase Tests (collision detection): **14/14 PASS**

All broad phase collision detection algorithms passed:
- NxN broadphase (multiple worlds, shape flags, edge cases)
- SAP broadphase (segmented + tile variants, shape flags, edge cases)
- Explicit pairs broadphase

### Root causes of failures/errors:

| Error | Count | Cause |
|-------|-------|-------|
| `No module named 'mujoco'` | ~540 | MuJoCo backend not fully installed |
| `No module named 'pxr'` | ~44 | USD/OpenUSD not available |
| `No module named 'trimesh'` | ~33 | trimesh not installed at test time |
| `No module named 'git'` | ~18 | GitPython not installed at test time |
| `No module named 'scipy'` | ~7 | scipy not installed at test time |
| `No module named 'torch'` | — | skipped (expected) |
| Simulation result check | ~36 | Numerical failures in example tests |

## Throughput Benchmark (GPU, XPBD Solver)

Sphere bodies falling onto a ground plane, CUDA capture warmup (20 steps), then 500 measured steps:

| Body Count | Throughput |
|------------|------------|
| 10 bodies  | 751.9 steps/sec |
| 50 bodies  | 750.8 steps/sec |
| 100 bodies | 757.7 steps/sec |
| 500 bodies | 745.1 steps/sec |

Throughput is roughly constant across body counts (GPU is not saturated at these sizes).

## Success Criteria Status

- [x] Newton installs without errors
- [x] Basic rigid body simulation runs (falling box settles at correct height)
- [x] Test suite runs — partial results documented (full suite crashed process pool)
- [x] Report pushed to `autopilot-reports/newton-test/`

## Benchmark Script

See `autopilot-reports/newton-test/newton_benchmark.py` for the reproducible benchmark.
