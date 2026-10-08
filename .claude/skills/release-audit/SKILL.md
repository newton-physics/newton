---
name: release-audit
description: "Use when auditing Newton's pending Towncrier fragments for keep/defer decisions, reviewing an RC for readiness, or calibrating against an already-shipped release."
disable-model-invocation: true
argument-hint: "[target-version]"
allowed-tools: Bash(git log *) Bash(git show *) Bash(git grep *) Bash(git tag *) Bash(git rev-parse *) Bash(git diff *) Bash(git ls-tree *) Bash(uvx --from towncrier==25.8.0 towncrier build --draft *) Bash(uv run --no-project python .agents/skills/release-audit/scripts/list_commits.py *) Bash(uv run --no-project python .agents/skills/release-audit/scripts/license_audit.py *) Bash(uv run --no-project python .agents/skills/release-audit/scripts/cleanup_report.py *) Bash(gh --version) Bash(gh auth status) Bash(gh gist create *) Bash(gh gist list *) Bash(gh gist view *) Bash(gh gist edit *) Bash(gh issue view *) Bash(gh issue list *) Read Write Grep Glob
---

# Release Audit

From the repository root, read and follow the canonical workflow at
`.agents/skills/release-audit/SKILL.md`. If it is unavailable, stop without
starting the audit.
