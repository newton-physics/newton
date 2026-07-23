---
name: release-changelog
description: Use when auditing or consolidating Newton changelog fragments, promoting Unreleased for a release, or reconciling a released changelog back to main.
---

# Newton Release Changelog

Treat `CHANGELOG.md` as the durable accumulator. Pending user-facing changes
are the union of its `[Unreleased]` section and unconsumed `.md` files under
`changelog.d/`. Dated sections are immutable release history.

## Audit pending changes

1. Protect released history. Diff `CHANGELOG.md` from the latest stable tag and
   require explicit maintainer approval for changes to existing dated sections.
2. Identify the release ref and comparison base. Audit the release branch once
   it exists; otherwise audit the intended main ref.
3. Validate the checked-out accumulator and render a non-mutating preview:
   ```bash
   uv run --no-project python scripts/changelog.py validate
   uv run --no-project python scripts/changelog.py build --dry-run
   ```
   The preview combines existing `[Unreleased]` entries with pending fragments.
4. Compare the preview with the release audit and commit range from the previous
   release. Confirm `.skip` files contain credible reasons.
5. Preserve information. Rephrase, split, merge, or recategorize only when the
   facts remain intact. Ask before deleting information or downgrading a
   user-visible change.
6. Use only `Added`, `Changed`, `Deprecated`, `Removed`, and `Fixed`, in that
   order. Keep migration and retesting guidance in affected entries.
7. Remove exact and semantic duplicates. When a feature and its fix both land
   in one cycle, describe the final user-visible behavior once.
8. Keep `Added` for new public APIs, options, features, examples, and docs. Put
   existing-API behavior, warning, default, importer, and solver changes in
   `Changed`, even when they expand support.
9. Add compact `(#NNNN)` references only after verifying the PR introduced the
   change. Prioritize breaking/default changes, public API additions,
   deprecations, removals, and major fixes.
10. Give every breaking, removed, deprecated, or default-changing entry a
    concrete action. Never direct users to `newton._src`.

Generated `<!-- changelog-fragment: ... -->` comments preserve fragment
identity while entries are accumulated under `[Unreleased]`. They are invisible
in rendered Markdown. Preserve them when editing accumulated entries; never
invent or reassign them manually. Commit any editorial accumulator changes
before promotion. The final promotion strips the comments from the dated public
section, and reconciliation recovers available IDs from the committed state
immediately before promotion while retaining exact-text compatibility for
legacy entries.

## Consolidate during development

Consolidation is safe at any time on `main` or a release branch:

```bash
uv run --no-project python scripts/changelog.py build --dry-run
uv run --no-project python scripts/changelog.py build
```

`build` merges pending fragments into `[Unreleased]`, deduplicates exact
entries, consumes `.md` and `.skip` files, and leaves all dated sections
unchanged. Commit the result in a changelog-only PR labeled
`changelog-maintenance`. Later pull requests continue adding fragments, and a
later build extends the existing accumulator.

## Promote the final release

On the release branch, preview and then promote the full accumulator for GA:

```bash
uv run --no-project python scripts/changelog.py release \
  --version X.Y.Z --date YYYY-MM-DD --dry-run
uv run --no-project python scripts/changelog.py release \
  --version X.Y.Z --date YYYY-MM-DD
```

`release` includes both previously consolidated entries and any remaining
fragments, resets `[Unreleased]`, inserts the dated section above older
releases, strips internal provenance comments from that public section, and
consumes pending fragments. Review the complete diff in a changelog-only PR
labeled `changelog-maintenance`.

## Reconcile to main

After tagging, create a changelog-only branch from current `main` and run:

```bash
uv run --no-project python scripts/changelog.py reconcile \
  --source-ref vX.Y.Z --version X.Y.Z --dry-run
uv run --no-project python scripts/changelog.py reconcile \
  --source-ref vX.Y.Z --version X.Y.Z
```

The command imports the exact tagged release, removes released entries from
main's `[Unreleased]` accumulator and pending fragments, consolidates remaining
main-only fragments, and preserves post-cut entries for the next release. Open
the result as a changelog-only PR labeled `changelog-maintenance`.

Do not replace `CHANGELOG.md` with the release-branch copy or blindly
cherry-pick a promotion commit over a diverged main accumulator.

## Checks

```bash
uv run --no-project python scripts/changelog.py validate
git diff v<latest-release> -- CHANGELOG.md changelog.d
git diff --name-status -- CHANGELOG.md changelog.d
rg -n "removed|deprecated|in favor of|use .* instead|renam|replac|default|breaking" \
  CHANGELOG.md changelog.d
```

Confirm that `[Unreleased]` is complete, no dated history changed, released
entries appear exactly once, and post-cut main entries survive reconciliation.
