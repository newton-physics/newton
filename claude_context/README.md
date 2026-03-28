# CENIC Implementation Log

Index of all tested approaches. Read this before proposing new work.

## Scaling

| Note | Commit | Status |
|---|---|---|
| [Sync-based step_dt](scaling/2026-03-13-sync-based-step-dt.md) -- fixed CUDA graph + .numpy() boundary check | `c2ec695` | abandoned |
| [K=8 flat graph](scaling/2026-03-13-k8-flat-graph.md) -- unrolled 8 iterations, no loop | `84e2877` | abandoned |
| [capture_while](scaling/2026-03-27-capture-while.md) -- wp.capture_while, zero CPU syncs. N^0.094 | `2d6f995` | **current** |
| [Direct subroutine calls](scaling/2026-03-27-direct-subroutine-calls.md) -- bypass mujoco_warp.step() | reverted | failed |
| [4-way comparison](scaling/2026-03-28-python-loop-vs-graph.md) -- fresh solvers, only capture_while scales | uncommitted | completed |

## Solver

| Note | Commit | Status |
|---|---|---|
| [Initial CENIC solver](solver/2026-03-01-initial-cenic-solver.md) -- step doubling, RMS error | `3380670` | partial |
| [Architecture overhaul](solver/2026-03-12-architecture-overhaul.md) -- step_dt, Drake controller | `cc15bed` | success |
| [Drake error controller](solver/2026-03-13-drake-error-controller.md) -- CalcAdjustedStepSize, inf-norm | `cc15bed`, `c0273b9` | **current** |
| [CUDA graph capture](solver/2026-03-13-cuda-graph-capture.md) -- ScopedCapture, timestep-pointer fix | `84e2877`, `9e3ca18` | success |
| [Error metric evolution](solver/2026-03-19-error-metric-evolution.md) -- RMS -> L2 -> inf-norm(q) | `b1d2148`, `c0273b9` | **current** |
| [Zero-alloc step](solver/2026-03-20-zero-alloc-step.md) -- StepWorkspace for graph capture | .venv patch | success |

## Contact

| Note | Commit | Status |
|---|---|---|
| [Benchmark infrastructure](contact/2026-03-12-benchmark-infrastructure.md) -- scenes, bench platform | `cc15bed`, `c0273b9` | success |

## Viewer

(no entries yet)
