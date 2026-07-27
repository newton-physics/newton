---
name: release-changelog
description: Use when auditing Newton changelog fragments, building a dated release changelog, or synchronizing a release build back to main.
---

# Newton Release Changelog

Pending user-facing changes live in Towncrier fragments under `changelog.d/`.
`CHANGELOG.md` keeps immutable dated history and is generated only on a release
branch. Follow `changelog.d/README.md` as the command and format authority.

## Audit pending changes

1. Identify the release ref and comparison base. Audit `release-X.Y` once it
   exists; otherwise audit the intended main ref.
2. Protect released history. Diff `CHANGELOG.md` from the latest stable tag and
   require explicit maintainer approval for edits to dated sections.
3. Validate fragments and render a non-mutating preview:
   ```bash
   uv run --no-project python scripts/changelog_policy.py validate
   uvx --from towncrier==25.8.0 towncrier build --draft \
     --version X.Y.Z --date YYYY-MM-DD
   ```
4. Compare the preview with the release audit and commit range from the previous
   release. Inspect `.skip` reasons separately.
5. Preserve information. Rephrase, split, merge, or recategorize fragments only
   when the facts remain intact. Ask before deleting information or downgrading
   a user-visible change.
6. Use only `Added`, `Changed`, `Deprecated`, `Removed`, and `Fixed`, in that
   order. Keep migration and retesting guidance in affected entries.
7. Remove exact and semantic duplicates. When a feature and its fix both land
   in one cycle, describe the final user-visible behavior once.
8. Keep `Added` for new public APIs, options, features, examples, and docs. Put
   existing-API behavior, warning, default, importer, and solver changes in
   `Changed`, even when they expand support.
9. Give every breaking, removed, deprecated, or default-changing entry a
   concrete action. Never direct users to `newton._src`.
10. A numeric fragment identifier is a GitHub issue number. Towncrier renders
    its issue link automatically; do not rewrite it as a pull request number.

## Build the final release

Build only on `release-X.Y`, after the release audit and final cherry-picks:

```bash
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
uvx --from towncrier==25.8.0 towncrier build --yes \
  --version X.Y.Z --date YYYY-MM-DD
git rm --ignore-unmatch "changelog.d/*.skip"
git add -A CHANGELOG.md changelog.d
```

Review and approve the draft before running the mutating command. Towncrier
inserts the dated section below `[Unreleased]` and deletes rendered fragments.
It ignores `.skip` files, so remove those explicitly. Review the staged diff in
a changelog-only pull request labeled `release-management`.

The first Towncrier release requires one migration audit. The insertion marker
sits above the legacy `[Unreleased]` entries so they remain under the first
generated release title. Merge duplicate category headings without dropping or
duplicating an entry. Later releases need no special handling.

## Synchronize to main

After tagging:

1. Create a changelog-only branch from current `main`.
2. Cherry-pick the exact Towncrier build commit from `release-X.Y`.
3. Confirm fragments deleted by the release disappear while fragments added to
   `main` after the branch cut remain under `changelog.d/`.
4. Confirm the dated section matches the release tag and older history is
   unchanged.
5. Open a changelog-only pull request labeled `release-management`.

Do not replace the whole file with the release-branch copy. The build commit's
path-level deletions are what preserve main-only fragments.

## Checks

```bash
uv run --no-project python scripts/changelog_policy.py validate
git diff v<latest-release> -- CHANGELOG.md changelog.d
git diff --name-status -- CHANGELOG.md changelog.d
rg -ni "removed|deprecated|in favor of|use .* instead|renam|replac|default|breaking" \
  CHANGELOG.md changelog.d
```

Confirm that `[Unreleased]` is empty after the first migration, no dated history
changed, released entries appear exactly once, and post-cut main fragments
survive synchronization.
