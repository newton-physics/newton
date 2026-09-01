---
name: release-changelog
description: Use when auditing Newton changelog fragments, building a dated release changelog, or synchronizing a release build back to main.
---

# Newton Release Changelog

Pending user-facing changes live in Towncrier fragments under `changelog/`.
`CHANGELOG.md` is generated only on a release branch. Shipped sections are
immutable; the assembled section for the pending release remains a rolling
document until tagging. Follow `changelog/README.md` as the command and format
authority.

## Audit pending changes

1. Identify the release ref and comparison base. Audit `release-X.Y` once it
   exists; otherwise audit the intended main ref.
2. Protect released history. Diff `CHANGELOG.md` from the latest stable tag and
   require explicit maintainer approval for edits to dated sections.
3. Render a non-mutating preview, which also validates Towncrier's renderable
   fragment filenames:
   ```bash
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

## Assemble the release during RC stabilization

After the initial release scope has been audited on `release-X.Y`, assemble the
current fragments early enough for maintainer review:

```bash
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
uvx --from towncrier==25.8.0 towncrier build --yes \
  --version X.Y.Z --date YYYY-MM-DD
git rm --ignore-unmatch "changelog/*.skip"
git add -A CHANGELOG.md changelog
```

Review and approve the draft before running the mutating command. Towncrier
inserts the dated section below `[Unreleased]` and deletes rendered fragments.
It ignores `.skip` files, so remove those explicitly. Review the staged diff in
a changelog-only pull request labeled `release-management`.

After assembly, apply the audit rules above to the dated section: verify
completeness, grouping, deduplication, wording, categories, and migration
guidance. Keep editorial cleanup in the changelog-management commits that will
later be synchronized to `main`.

The first Towncrier release requires one migration audit. The insertion marker
sits above the legacy `[Unreleased]` entries so they remain under the first
generated release title. Merge duplicate category headings without dropping or
duplicating an entry. Later releases need no special handling.

Treat the assembled section as a rolling document. For every later cherry-pick
before tagging:

1. Validate the new fragments and render them with `towncrier build --draft`.
2. Fold the previewed entries into the existing dated section without creating
   a second release heading.
3. Delete exactly the consumed `.md` and `.skip` fragments, then stage
   `CHANGELOG.md` and `changelog/`.
4. Rerun the changelog cleanup and `release-audit` checks, and merge the update
   as another changelog-only `release-management` pull request.

Final GA preparation verifies the completed section and confirms that no
release-branch fragments remain. Do not postpone the full cleanup until GA.

## Synchronize to main

After tagging:

1. Create a changelog-only branch from current `main`.
2. Cherry-pick, in order, every changelog-management commit from `release-X.Y`:
   the initial Towncrier build, editorial cleanup, and all later cherry-pick
   additions.
3. Confirm fragments deleted by those commits disappear while fragments added to
   `main` after the branch cut remain under `changelog/`.
4. Confirm the dated section matches the release tag and older history is
   unchanged.
5. Open a changelog-only pull request labeled `release-management`.

Do not replace the whole file with the release-branch copy. The commits'
path-level deletions are what preserve main-only fragments.

## Checks

```bash
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
git diff v<latest-release> -- CHANGELOG.md changelog
git diff --cached --name-status -- CHANGELOG.md changelog
rg -ni "removed|deprecated|in favor of|use .* instead|renam|replac|default|breaking" \
  CHANGELOG.md changelog
```

Confirm that `[Unreleased]` is empty after the first migration, no dated history
changed, released entries appear exactly once, and post-cut main fragments
survive synchronization.
