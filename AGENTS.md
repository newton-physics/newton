# Newton Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing Newton code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst) and follow `.claude/skills/code-review-newton/SKILL.md`.

- Create a feature branch on your fork before committing—never commit directly to `main`. Give the pull request a concise, descriptive title.
- Use imperative mood in commit messages ("Fix X", not "Fixed X"), with a roughly 50-character subject and a body wrapped at 72 characters that explains what and why.
- Verify regression tests fail without the fix before committing.

Run `uvx pre-commit run -a` to lint and format before committing. Use `uv` for all commands; fall back to `venv` or `conda` only if `uv` is unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Collab-fork solver branches (skild workstream)

This fork's solver work is consumed as a **git submodule of `skild-ai/skild-IL-solver`**,
whose production branch (`fpgs-main`) pins exact commits of the branches below. That
changes the branch rules:

- Lineage (full map: `reports/vishal/robotiq/branches.md` in skild-IL-solver):
  base `3ee2ff96` → `vishal/fpgs-mimic-rows` (FPGS mimic/connect rows, passive joint
  springs, CRBA loop-joint indexing fix) → `vishal/fpgs-preelim` (bilateral
  pre-elimination). `vishal/kamino-robotiq` is a pinned workspace at the mimic-rows tip.
  New solver features branch off the current validated tip (`<username>/<feature-desc>`
  per the rule above).
- **Push before pointer.** Push the newton-collab branch to origin *before* committing
  the parent-repo submodule pointer that references it — otherwise fresh parent
  checkouts fail at submodule init.
- **Never force-push or rewrite a branch whose commits a parent pointer references.**
  Remote GC can prune orphaned SHAs and break every parent checkout pinned to them.
  When splitting history for upstream PRs (`git rebase --onto origin/main 3ee2ff96`),
  create *new* branches and leave the referenced ones intact.
- **Validation gates before a tip may become a parent pointer**: the FPGS test modules
  pass (`uv run --extra dev python -m unittest newton.tests.test_feather_pgs_mimic
  newton.tests.test_feather_pgs_connect newton.tests.test_feather_pgs_springs
  newton.tests.test_feather_pgs_preelim`), regression-first discipline holds (tests
  verified failing without the change), and solver changes that affect grasp behavior
  pass the parent repo's P25 sustained-squeeze gate
  (`P25_FPGS_PREELIM=1 P25_FPGS_ITERS=64 ... p25_curling_grasp.py --solver feather_pgs
  --object pinch --headless --steps 300 --close-at 60 --rigid-gap 0.005` must hold the
  cube).
- **Warp kernel cache when bisecting generated-source edits** (e.g. the FPGS sweep-phase
  filters): stale caches in `~/.cache/warp` silently run the old kernels — clear the
  cache between bisection steps.

## Tests

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Follow `changelog/README.md`: add a Towncrier fragment for user-facing changes instead of editing `CHANGELOG.md` directly. A `.skip` reason is optional for changes without user-facing impact.
- Preview fragments with `uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD`.

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` or `test_post_step()`; an example may implement both.
  - In test mode, `test_post_step()` runs after each simulation step and `test_final()` runs after the example completes.
- Register the example in `README.md` with its `python -m newton.examples <name>` command and a 320x320 JPEG screenshot.
