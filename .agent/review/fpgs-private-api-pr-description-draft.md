# Draft PR Description

## Summary

Simplify the private FeatherPGS API line to the current matrix-free path.

This branch stops presenting the private solver as a bundle of retained
ablations. Instead, it bakes in the current matrix-free contact solve path and
removes obsolete top-level mode selection and kernel-selection API knobs from
`SolverFeatherPGS`.

## What Changed

- Remove the private solver constructor knobs for `pgs_mode` and per-stage
  kernel selection.
- Run the private FeatherPGS step path as matrix-free only.
- Add focused unit coverage for the stripped-down constructor surface and a
  minimal step smoke test.

## Why Matrix-Free Only

The published nightly ablations already show that the matrix-free path is the
 winner for the private line, while the dense/split modes mostly preserve
 research history rather than current product intent.

Using the published nightly run `2026-04-01T20-49-30Z` (`summary.json`,
commit `53b3188`) from the `gh-pages` artifacts:

| Scenario | Hardware | Baseline | Split | Matrix-free | Matrix-free vs split |
| --- | --- | ---: | ---: | ---: | ---: |
| `h1_tabletop_ablation` | RTX 5090 | 4,262 env_fps | 41,496 env_fps | 118,547 env_fps | 2.86x |
| `h1_tabletop_ablation` | RTX PRO 6000 Server | 4,942 env_fps | 47,528 env_fps | 114,531 env_fps | 2.41x |
| `h1_tabletop_ablation` | B200 | 3,765 env_fps | 60,684 env_fps | 107,262 env_fps | 1.77x |

The same nightly run also supports keeping the current winner kernel choices on
the private line instead of exposing them as API:

| Scenario | Hardware | FeatherPGS baseline | tiled `hinv_jt` | tiled PGS | parallel streams |
| --- | --- | ---: | ---: | ---: | ---: |
| `g1_flat_ablation` | RTX 5090 | 590,152 env_fps | 1,358,725 env_fps | 1,461,194 env_fps | 1,461,373 env_fps |
| `g1_flat_ablation` | RTX PRO 6000 Server | 504,100 env_fps | 1,182,672 env_fps | 1,276,901 env_fps | 1,277,128 env_fps |
| `g1_flat_ablation` | B200 | 760,212 env_fps | 1,389,444 env_fps | 1,551,859 env_fps | 1,550,952 env_fps |

Interpretation:

- `matrix-free` materially outperforms `split` on the published tabletop
  ablation across all listed GPUs.
- `tiled hinv_jt`, tiled PGS, and parallel streams are already the winning
  direction in the published flat-scene ablation, so the private API does not
  need to keep exposing these as branch-local tuning knobs.

## Validation

- `uv run --extra dev -m newton.tests -k test_feather_pgs`
- `uvx pre-commit run -a`

## Notes For Review

- This PR intentionally updates only the private API line. It does not touch
  `gh-pages` content and does not move benchmark rationale into public docs.
- The benchmark figures above are copied from published nightly artifacts
  already present on `origin/gh-pages`, not from new benchmark runs in this
  branch.
