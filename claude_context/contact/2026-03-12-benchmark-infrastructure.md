---
aliases:
  - benchmark infrastructure
  - bench platform
date: 2026-03-12
status: success
commit: cc15bed
tags:
  - contact
  - benchmarking
  - success
---

# Contact benchmark infrastructure

## Scene: `scripts/scenes/contact_objects.py`

Drops rigid bodies onto ground plane with varying contact complexity. Shared library: `build_model()`, `make_solver()`, `make_fixed_solver()`.

## Benchmark platform: `scripts/bench/`

```
uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20
```

- `benchmarks/scaling.py` -- N-scaling (graph/fixed/manual), 5 plots
- `benchmarks/components.py` -- per-kernel breakdown
- `benchmarks/accuracy.py` -- wall-vs-tol, error/dt traces
- Fresh solver per (N, mode) -- fixes state contamination
- Results versioned by git commit hash in `results/<hash>/`

## Key lesson

> [!important] Fresh solver per measurement
> Without this, CUDA context state (graph caches, allocations) leaks between runs, producing unreliable timings. State contamination was the root cause of multiple misleading benchmark results.

Related: [[2026-03-28-sync-based-measured|sync-based step_dt]], [[2026-03-28-world-divergence-root-cause|world divergence]].
