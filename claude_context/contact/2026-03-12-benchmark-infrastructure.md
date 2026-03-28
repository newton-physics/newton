# Contact Benchmark Infrastructure

**Date:** 2026-03-12
**Area:** contact
**Status:** success
**Commit:** cc15bed (initial), c0273b9 (major overhaul)
**Tags:** #contact #success

## Goal
Build benchmark infrastructure for testing CENIC adaptive stepping against contact-rich scenes: a test scene, automated benchmark runs, and scaling diagnostics.

## Approach

### Scene: scripts/scenes/contact_objects.py (was testing/contact/cenic_contact_objects.py)
Primary test scene. Drops rigid bodies onto ground plane with varying contact complexity.
- Shared library: `build_model()`, `make_solver()`, `make_fixed_solver()`
- Interactive demo moved to `scripts/demos/contact_objects.py`

### Benchmark platform: scripts/bench/ (was testing/contact/cenic_benchmark_plots.py + cenic_kernel_scaling.py)
Reorganized 2026-03-28 into a proper platform:
- Entry point: `uv run -m scripts.bench`
- `benchmarks/scaling.py` -- N-scaling (graph/loop/fixed/manual), 4 plots
- `benchmarks/components.py` -- per-kernel breakdown with sync barriers
- `benchmarks/accuracy.py` -- wall-vs-tol, error/dt traces (subprocess isolation)
- Fresh solver per (N, mode) -- fixes state contamination bug found 2026-03-28
- Results versioned by git commit hash in `results/<hash>/`

### Archived: scripts/archive/
- `scaling_diag.py` -- per-component timing (pre-capture_while architecture, stale)
- `diag_iteration_spike.py` -- iteration spike diagnostic (stale)

## Results
Infrastructure operational. Contact objects scene successfully exercises adaptive stepping with varying contact density. Benchmark plots show expected behavior: dt drops during contact events, error stays below tol, wall time scales sub-linearly with N.

## Verdict
The subprocess isolation in benchmark_plots was essential -- without it, CUDA context state from one run (graph caches, allocations) leaked into the next, producing unreliable timings. The scaling_diag script is obsolete for the current architecture but could be adapted to measure graph-level timings.
