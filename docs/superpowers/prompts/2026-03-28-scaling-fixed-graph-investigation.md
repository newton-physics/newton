# CENIC: Measure scaling of approach #1 (sync-based step_dt)

## Problem

Row #1 in the N-scaling optimization log (CLAUDE.md) says "NOT MEASURED". We replaced that approach with capture_while without ever measuring its scaling. This session fills in that row. Nothing else.

## Steps

1. Read the optimization log table in CLAUDE.md
2. Check out c2ec695 in a git worktree
3. Copy `scripts/testing/contact/cenic_kernel_scaling.py` from main into the worktree
4. Adapt the script if the c2ec695 solver API differs (it will -- read the solver file)
5. Run: `--mode graph-vs-manual --ns 1 4 16 64 256 --steps 50 --warmup 20`
6. Record results in `docs/superpowers/reports/2026-03-28-sync-step-dt-scaling.md`:
   - What the implementation actually was (copy the relevant step_dt code)
   - Graph exponent, manual exponent, absolute times
   - Comparison table against rows #3 and #4
7. Update row #1 in CLAUDE.md with the measured numbers and verdict

## Rules

- DO NOT propose fixes
- DO NOT refactor anything
- DO NOT start a new approach
- Just measure and report
