# FeatherPGS Private API Matrix-Free ExecPlan

## Goal

Create a stripped-down private FeatherPGS API branch that keeps only the current
matrix-free winner path, removes obsolete top-level and kernel-level branching,
and leaves a reviewable PR-description draft in the workspace instead of editing
the GitHub PR description directly.

## Branch and Workspace

- Working branch: `dturpin/fpgs-private-api-matrix-free`
- Tracking branch: `origin/fpgs-private-api`
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
- Remaining cleanup for later passes: remove dead dense/split-only helper code
  and supporting branch-local references that are no longer reachable.

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

Required work:
- Re-run the final focused test set.
- Re-run `uvx pre-commit run -a`.
- Update this ExecPlan to reflect what shipped and any intentional omissions.

Validation:
- Record exact commands run and results.

Checkpoint:
- Commit and push the final polishing pass.
- Stop and wait for human review.

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
- If a milestone is too large to finish cleanly, begin the pass by tightening
  this ExecPlan with a short replan note and then complete one reviewable slice
  of that milestone.
- Never update the real PR description in GitHub for this task.
- Never touch `gh-pages` for this task.
- Prefer borrowing ideas or code from `origin/feather_pgs` surgically instead of
  trying to merge the whole research branch into the private API line.
