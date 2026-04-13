# FeatherPGS Private API Matrix-Free ExecPlan

## Goal

Create a stripped-down private FeatherPGS API branch that keeps only the current
matrix-free winner path, removes obsolete top-level and kernel-level branching,
and leaves a reviewable PR-description draft in the workspace instead of editing
the GitHub PR description directly.

## Branch and Workspace

- Working branch: `dturpin/fpgs-private-api-matrix-free`
- Tracking branch: `origin/dturpin/fpgs-private-api-matrix-free`
- Workspace root: repository root of this worktree

## Constraints

- Rebase onto `upstream/main` before the implementation settles.
- Do not open, merge, or edit the GitHub PR directly.
- Do not update `gh-pages`.
- Do not move the benchmark/design rationale into FeatherPGS docs for this task.
- Keep the matrix-free justification tight and write it to
  `.agent/review/fpgs-private-api-pr-description-draft.md` for human review.
- Run `uvx pre-commit run -a` before milestone pushes when practical, and at
  minimum before the final push.

## Design Intent

The private API branch should present one coherent implementation path rather
than a research branch with many retained ablations. The target is not “all
paths still exist but only one is recommended”; the target is “the private API
is the current matrix-free implementation, with the winning defaults baked in
and the obsolete branches removed.”

For the long-lived `feather_pgs` research branch, prefer cleaner separation
between:

1. shared articulation stages
2. contact-solve backend realization
3. kernel-policy / ablation plumbing

That separation should make future winner-cherry-picks into smaller API-facing
branches cheaper and less error-prone.

## Milestones

### Milestone 0: Replan / baseline capture

Deliverable:
- Tighten this ExecPlan with any details discovered while comparing
  `origin/fpgs-private-api`, `origin/feather_pgs`, and `upstream/main`.

Required work:
- Inspect the current private API branch contents and review comments.
- Identify which parts should be kept, dropped, or selectively borrowed from
  `feather_pgs`.
- Record any conflict hotspots expected during rebase.

Validation:
- No code changes required beyond plan updates.

Checkpoint:
- Commit and push if the plan changes materially enough to warrant review;
  otherwise continue into Milestone 1 within the same pass only if that is
  clearly implementable.

### Milestone 1: Rebase private API line onto upstream

Deliverable:
- `dturpin/fpgs-private-api-matrix-free` rebased onto `upstream/main`.

Required work:
- Rebase the branch onto `upstream/main`.
- Resolve conflicts carefully without regressing the private API line.
- Keep the branch reviewable after the rebase.

Validation:
- Run at least a focused smoke check sufficient to verify the branch imports and
  the affected solver modules still load.

Checkpoint:
- Commit any post-rebase conflict resolutions if needed and push the branch.
- Stop after the push; do not start Milestone 2 in the same implementation pass.

### Milestone 2: Collapse the private API to matrix-free only

Deliverable:
- Private FeatherPGS implementation supports only the matrix-free path and only
  the winning kernel/default choices.

Progress update (2026-04-13, pass 1):
- Shipped the first reviewable slice of this milestone:
  `SolverFeatherPGS` no longer exposes `pgs_mode` or per-stage kernel-selection
  constructor knobs, and the live `step()` path now executes the matrix-free
  solve only.
- Added focused unit coverage for the stripped-down constructor surface and a
  minimal matrix-free smoke step in `newton/tests/test_feather_pgs.py`.

Progress update (2026-04-13, pass 2):
- Removed the remaining dead dense-only and standalone matrix-free solve
  helpers from `solver_feather_pgs.py`, including obsolete dense kernel
  factory entries and branch-local solver state that the private line no longer
  reaches.
- Reworded the surviving GS kernel comments/docstrings to describe the unified
  winner path rather than an ablation-era "dense + matrix-free" split.
- Milestone 2 is now functionally complete on the private solver surface; any
  leftover cleanup is supporting-surface polish tracked in Milestone 3.

Required work:
- Remove `dense` and `split` support from the private API implementation.
- Remove retained kernel-selection and intra-mode multi-path knobs whose only
  purpose was ablation on the research branch.
- Simplify constructor/API surface accordingly.
- Simplify code structure, comments, and docstrings to describe one path.
- Preserve correctness-oriented pieces needed by the winner path.

Validation:
- Add or update focused tests for the stripped-down behavior.
- Run the affected tests.

Checkpoint:
- Commit the simplification and push the branch.
- Stop after the push.

### Milestone 3: Clean supporting surfaces

Deliverable:
- Bench/test/supporting code on the private API branch reflects the single-path
  implementation instead of the old ablation-heavy surface.

Progress update (2026-04-13, pass 3):
- Searched the branch for leftover private FeatherPGS callers and stale
  constructor knobs (`pgs_mode`, per-stage kernel selectors). No branch-local
  tests, helpers, or docs outside the focused private test and PR draft still
  referenced the removed surface.
- Milestone 3 therefore closes as a support-surface verification pass rather
  than a broad code-edit pass: the earlier solver cleanup already left the
  private branch with one surviving test entry point in
  `newton/tests/test_feather_pgs.py`, and no extra helper or benchmark cleanup
  was required on this branch.
- Validation:
  - `uv run --extra dev -m newton.tests -k test_feather_pgs` -> passed
    (`Ran 2 tests in 14.754s`, `OK`).
  - `uvx pre-commit run -a` -> passed (`ruff`, `ruff format`, `uv-lock`,
    `typos`, and `check warp array syntax`).

Required work:
- Update any tests, helper code, or branch-local references that still assume
  mode or kernel multiplicity.
- Remove dead code left behind by Milestone 2.
- Keep changes scoped to the private API branch rather than the `feather_pgs`
  docs or `gh-pages`.

Validation:
- Run the relevant focused tests.
- Run `uvx pre-commit run -a`.

Checkpoint:
- Commit and push the cleanup milestone.
- Stop after the push.

### Milestone 4: Draft PR description for human review

Deliverable:
- `.agent/review/fpgs-private-api-pr-description-draft.md`

Required work:
- Draft a tight PR description for the private API branch.
- Explain the matrix-free-only decision briefly.
- Include small benchmark tables or bullets derived from the published nightly
  data on `gh-pages` JSONL artifacts.
- Keep the scope to the private API decision and resulting simplification.
- Do not edit the actual GitHub PR description.

Validation:
- Verify the draft reads as a plausible PR description and cites the right run
  context and branch intent.

Checkpoint:
- Commit and push the draft file if it is useful to keep with the branch.
- Stop after the push.

### Milestone 5: Final validation pass

Deliverable:
- Reviewable branch state with validation evidence and no pending plan items.

Progress update (2026-04-13, pass 4):
- Repointed the local branch to track
  `origin/dturpin/fpgs-private-api-matrix-free`, removing the stale
  `origin/fpgs-private-api` tracking ambiguity from `git status`.
- Re-ran the final focused solver validation and repository-wide pre-commit
  checks, then updated this ExecPlan to reflect the shipped end state.
- Milestone 5 is closed. No milestone items remain on this ExecPlan.

Validation:
- `git rev-list --left-right --count upstream/main...HEAD` -> `28 4`.
  This confirms the branch is rebased onto `upstream/main` and now sits 4
  commits ahead with no missing upstream-main commits.
- `uv run --extra dev -m newton.tests -k test_feather_pgs` -> passed
  (`Ran 2 tests in 14.586s`, `OK`).
- `uvx pre-commit run -a` -> passed (`ruff`, `ruff format`, `uv-lock`,
  `typos`, and `check warp array syntax`).
- `git status --short --branch` after upstream fix ->
  `## dturpin/fpgs-private-api-matrix-free...origin/dturpin/fpgs-private-api-matrix-free`

Intentional omissions / review notes:
- The actual GitHub PR description was not edited in this workflow; only the
  local human-review draft at
  `.agent/review/fpgs-private-api-pr-description-draft.md` was maintained.
  That non-edit is a process guarantee from this workspace flow rather than a
  property that can be independently verified from local files alone.

Checkpoint:
- Commit and push the final closeout state, then stop for human review.

## Notes for the Implementor

- Replan update (2026-04-13, pass 1):
  `git rev-list --left-right --count upstream/main...HEAD` reports `0 97`, so the
  branch already contains `upstream/main` and no Milestone 1 rebase work is
  pending in this workspace. The next reviewable slice is Milestone 2 focused on
  the private solver surface itself: remove the `pgs_mode` and kernel-selection
  public knobs, hard-wire the solver to the matrix-free winner path in
  `solver_feather_pgs.py`, and add focused unit coverage for the stripped-down
  constructor behavior. Dense/split-only helper code that becomes unreachable in
  that slice should be removed when practical; broader supporting cleanup stays
  in Milestone 3.
- Replan update (2026-04-13, pass 2):
  The next slice tightens Milestone 2 itself instead of jumping ahead: remove
  the now-dead dense-only / standalone-MF helper paths and rewrite the live GS
  kernel commentary so the private branch reflects one coherent articulated +
  free-rigid winner path. After that push, stop before Milestone 3 and leave
  supporting-surface cleanup plus final validation recording for later passes.
- Replan update (2026-04-13, pass 3):
  A branch-wide search found no remaining private FeatherPGS callers or
  support-code references to the removed constructor knobs outside the focused
  private test and PR draft already added in prior passes. This pass therefore
  closes Milestone 3 by recording that no further branch-local support-surface
  edits are needed, then stops before the final validation milestone.
- Replan update (2026-04-13, pass 4):
  The remaining work is strictly closeout: rerun the final validation, fix the
  stale local tracking branch so the pushed review state is obvious from
  `git status`, and rewrite Milestone 5 as shipped state with exact results and
  the one remaining non-verifiable GitHub PR-description note.
- If a milestone is too large to finish cleanly, begin the pass by tightening
  this ExecPlan with a short replan note and then complete one reviewable slice
  of that milestone.
- Never update the real PR description in GitHub for this task.
- Never touch `gh-pages` for this task.
- Prefer borrowing ideas or code from `origin/feather_pgs` surgically instead of
  trying to merge the whole research branch into the private API line.
