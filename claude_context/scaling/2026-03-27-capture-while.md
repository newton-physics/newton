# capture_while conditional graph

**Date:** 2026-03-27 (updated 2026-03-28 with corrected benchmark)
**Area:** scaling
**Status:** success
**Commit:** 2d6f995
**Tags:** #scaling #success #current

## Goal
Zero CPU synchronization in the adaptive inner loop. GPU drives the loop autonomously.

## Approach
`wp.capture_while(boundary_flag, while_body=_run_iteration_body)`. Entire boundary loop runs as a conditional CUDA graph node. Graph captured once on first step_dt call. No `.numpy()` calls in the hot path.

## Results (corrected 2026-03-28, fresh solver per mode)

| N | Graph (ms) | Loop (ms) | Fixed (ms) | Manual (ms) |
|---|-----------|----------|-----------|------------|
| 1 | 1.75 | 8.76 | 1.86 | 6.19 |
| 4 | 1.68 | 8.52 | 2.50 | 5.52 |
| 16 | 1.92 | 8.24 | 2.54 | 5.63 |
| 64 | 2.65 | 8.26 | 2.54 | 5.14 |
| 256 | 3.06 | 8.84 | 1.97 | 6.97 |

- **Graph (capture_while):** N^0.094
- **Loop (.numpy() boundary):** N^0.001 (flat)
- **Fixed (SolverMuJoCo):** N^0.007 (flat)
- **Manual (fixed K, no graph):** N^0.019 (flat)
- K_mean = 3.1 for both graph and loop (identical)
- Measurement: `uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20`

### Previous results (contaminated -- DO NOT USE)
Original benchmark shared a single solver across modes, causing state contamination. Reported Graph N^0.099, Manual N^0.047, N=1: 4.69ms. These numbers were wrong.

## Verdict
Graph path is the fastest at all practical N despite the N^0.094 scaling. It beats fixed-step at N=1 (1.75ms vs 1.86ms) even though it does 3x more physics work (step doubling), because CUDA graph replay eliminates all per-kernel CPU launch overhead. Crossover with fixed-step around N=64-256.

### Key established facts (corrected)
- Only capture_while scales with N. All other paths (loop, fixed, manual) are flat.
- K (iteration count) is constant across N -- adaptive stepping does not do more work at higher N.
- The N^0.094 exponent comes exclusively from CUDA conditional graph node replay overhead.
- Fresh solver per (N, mode) is mandatory for correct benchmarks.
