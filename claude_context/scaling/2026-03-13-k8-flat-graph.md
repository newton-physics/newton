# K=8 flat graph

**Date:** 2026-03-13
**Area:** scaling
**Status:** abandoned
**Commit:** 84e2877
**Tags:** #scaling #abandoned

## Goal
Avoid per-iteration overhead by unrolling a fixed number of iterations into a single CUDA graph.

## Approach
Fixed CUDA graph with K=8 iterations unrolled (no loop, no conditional). All 8 iterations baked into graph nodes.

## Results
NOT MEASURED. Abandoned in favor of capture_while.

## Verdict
No data recorded. The fixed K=8 means wasted work when fewer iterations suffice and insufficient iterations when more are needed. Inflexible, but the performance characteristics were never measured.
