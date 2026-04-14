---
name: newton-upstream-sync
description: Safely sync a Newton research fork with `newton-physics/newton` when updating local or fork `main`, rebasing `research/pressure-field`, handling dirty worktrees, preserving old branch tips on backup branches, validating hydro-pressure behavior after a rebase, or cleaning up local branch and worktree state afterward.
---

# Newton Upstream Sync

Keep `main` as a pure upstream integration branch. Keep pressure-field work on
`research/pressure-field` or another dedicated research branch.

## Clean Fast Path

Use the bundled PowerShell workflow when the current checkout is clean:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1
```

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -CreateResearchBranch
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -Strategy merge
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -ResearchBranch research/my-experiment
```

The script refuses to run on a dirty worktree unless `-AllowDirty` is passed.
Do not use `-AllowDirty` for branch surgery. Switch to the manual flow below.

## Manual Flow For Dirty Worktrees

If `git status --short --branch` is dirty, or the current checkout is on a
research branch with uncommitted work, do not rewrite branches in place.

1. Confirm remotes:

```bash
git remote -v
```

Expect `upstream` fetch to point at
`https://github.com/newton-physics/newton.git`.

2. Fetch upstream:

```bash
git fetch upstream
```

3. Preserve old fork tips on date-stamped backup branches before rewriting any
remote branch:

```bash
git branch main-pre-upstream-sync-YYYYMMDD origin/main
git push -u origin main-pre-upstream-sync-YYYYMMDD:main-pre-upstream-sync-YYYYMMDD

git branch research/pressure-field-pre-upstream-rebase-YYYYMMDD research/pressure-field
git push -u origin research/pressure-field-pre-upstream-rebase-YYYYMMDD:research/pressure-field-pre-upstream-rebase-YYYYMMDD
```

4. If the current checkout is dirty, update local `main` without checking it
out there. First verify it is a pure fast-forward:

```bash
git rev-list --left-right --count main...upstream/main
git merge-base --is-ancestor main upstream/main && echo fast-forward
```

If it is a clean fast-forward, move the ref directly:

```bash
git update-ref refs/heads/main refs/remotes/upstream/main <old-main-sha>
```

If it is not a fast-forward, stop and inspect the local `main` history before
rewriting anything.

5. Rewrite fork `main` only after preserving the old remote tip:

```bash
git push --force-with-lease=main:<old-origin-main-sha> origin main:main
```

## Rebase The Research Branch In A Clean Worktree

Never run the rebase inside the dirty original checkout. Create a detached temp
worktree from the research tip:

```bash
git worktree add --detach /tmp/newton-pressure-rebase research/pressure-field-pre-upstream-rebase-YYYYMMDD
git -C /tmp/newton-pressure-rebase switch -c tmp/research-pressure-rebase-YYYYMMDD
git -C /tmp/newton-pressure-rebase rebase main
```

If the rebase conflicts:

- Inspect conflict markers directly in the temp worktree.
- Compare the file on `main` with the final research branch state, not just the
  first conflicting commit:

```bash
git show main:path/to/file
git show research/pressure-field:path/to/file
```

- Prefer the final research-branch behavior when later pressure-field commits
  already refactor the same code.
- Continue with normal git commands:

```bash
git -C /tmp/newton-pressure-rebase add <resolved-files>
GIT_EDITOR=true git -C /tmp/newton-pressure-rebase rebase --continue
```

## Verification

For the pressure-field branch, run both a focused test slice and the example
smoke test against the rebased tree.

Prefer reusing the repo's existing `.venv` and point `PYTHONPATH` at the temp
worktree instead of creating a fresh environment in `/tmp`:

```bash
MPLCONFIGDIR=/tmp TMPDIR=/tmp PYTHONPATH=/tmp/newton-pressure-rebase \
  /mnt/d/Biomotions/newton/.venv/bin/python -m newton.tests \
  -p test_hydroelastic.py -k pressure --serial-fallback -q

MPLCONFIGDIR=/tmp TMPDIR=/tmp PYTHONPATH=/tmp/newton-pressure-rebase \
  /mnt/d/Biomotions/newton/.venv/bin/python -m \
  newton.examples.contacts.example_hydro_pressure_slice \
  --test --viewer null --num-frames 1 --shape box
```

Notes:

- The example is CUDA-only.
- Inside restricted sandboxes, the unittest runner may need escalation so
  `multiprocessing.Manager()` can bind locally.
- Warp must see the real CUDA driver for meaningful pressure-field validation.

## Publish The Rebases

After verification succeeds:

```bash
git branch research/pressure-field-rebased-YYYYMMDD <rebased-sha>
git push --force-with-lease=research/pressure-field:<old-origin-research-sha> \
  origin tmp/research-pressure-rebase-YYYYMMDD:research/pressure-field
```

## Move The Original Checkout Onto The Clean Branch

If the original checkout still contains branch-relative deletions or other
local changes, stash them before switching:

```bash
git stash push -m 'pre-switch dirty deletions before moving to rebased pressure branch YYYY-MM-DD'
```

Then:

```bash
git branch -m research/pressure-field research/pressure-field-local-pre-rebase-YYYYMMDD
git branch -m research/pressure-field-rebased-YYYYMMDD research/pressure-field
git branch --set-upstream-to=origin/research/pressure-field research/pressure-field
```

Leave the stash in place unless you intentionally want to reapply those old
changes.

## Cleanup

After the original checkout is on the clean rebased branch:

```bash
git branch -d tmp/research-pressure-rebase-YYYYMMDD
git worktree remove /tmp/newton-pressure-rebase
git worktree prune
```

If a temp worktree is broken and `git worktree remove` fails because its `.git`
file points at stale metadata, remove both the temp directory and the matching
`.git/worktrees/<name>` entry together.

You can optionally retarget local backup branches so their upstream tracking
matches the preserved backup remotes.
