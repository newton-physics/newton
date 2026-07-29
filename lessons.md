# Project Lessons

## 2026-07-29 — Keep the primary checkout aligned with the user's daily branch

- Context: Configuring this Newton fork as the editable source for Isaac Sim 6.0.1.
- Mistake: Kept `main` at `/home/limx/github/newton` and placed the actual development branch in a hidden `.worktrees/` path, adding unnecessary friction to the user's normal solver workflow.
- Rule: When the user only needs one compatibility branch for daily development, make the canonical repository path check out that branch. Keep other branches available remotely or as Git refs unless the user explicitly asks for simultaneous local checkouts.
