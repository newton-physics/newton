---
name: newton-upstream-sync
description: Sync a Newton research fork with the upstream `newton-physics/newton` repository while keeping pressure-field work isolated on a dedicated research branch. Use when updating `main`, rebasing or merging a research branch, or setting up a safe upstream workflow for this repo.
---

# Newton Upstream Sync

Use this skill when the goal is to keep `main` close to upstream Newton and keep pressure-field work isolated on a separate branch.

## Workflow

1. Keep upstream integration on `main`.
2. Keep pressure-field changes on `research/pressure-field` or another dedicated research branch.
3. Sync by running the bundled script from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1
```

By default the script:

- Fetches `upstream`
- Fast-forwards local `main` to `upstream/main`
- Rebases `research/pressure-field` onto `main`

## Common Variants

Create the research branch on first run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -CreateResearchBranch
```

Use merge instead of rebase when the research branch is shared:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -Strategy merge
```

Use a different research branch name:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_upstream.ps1 -ResearchBranch research/my-experiment
```

Generate a morning report after a nightly sync:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/nightly_sync_report.ps1
```

Register a nightly scheduled task at 2:00 AM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_nightly_sync_task.ps1
```

## Safety Rules

- The script refuses to run on a dirty worktree unless `-AllowDirty` is passed.
- `main` is updated with `git merge --ff-only` so it stays a clean upstream integration branch.
- Research work stays off `main`; pressure-field development should happen on the research branch.
- If the rebase or merge conflicts, resolve the files and continue with normal git commands.
- The nightly report skips sync if the worktree is dirty and writes a markdown report under `reports/upstream-sync/`.
