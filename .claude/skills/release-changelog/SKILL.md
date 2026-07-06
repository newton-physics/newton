---
name: release-changelog
description: Use when editing, auditing, or preparing Newton CHANGELOG.md for a release, especially to make upgrade-impact information actionable for developers.
---

# Newton Release Changelog

Maintain `CHANGELOG.md` as the detailed upgrade source of truth. Release notes
and release announcements carry the high-level summary; the changelog should
preserve specific breaking changes, removals, deprecations, behavior/default
changes, dependency constraints, and migration guidance.

## Workflow

1. Identify the release ref and comparison base. For final releases, use the
   final tag or release branch. For RC prep, use the latest RC tag as temporary
   ground truth and verify against the previous released tag.
2. Read the current `CHANGELOG.md` section being edited, the release audit if
   one exists, and PRs behind unclear entries. Do not rely only on commit
   subjects for migration guidance.
3. Preserve information. Rephrase, split, merge, and regroup entries only when
   the facts remain intact. Ask before deleting information, omitting a
   questionable entry, or downgrading a user-visible change to silence.
4. If there is no separate release-notes document, consider an upgrade-focused
   block near the top of the release section when the release has many changes.
   Use concise groups such as:
   - `Breaking Changes And Removals`
   - `Behavior And Default Changes To Re-check`
   - `New Deprecations To Plan Around`
5. If release notes already exist for the same release, avoid adding another
   summary block to `CHANGELOG.md`. Keep the changelog detail-oriented by
   improving entries in the canonical Keep-a-Changelog categories (`Added`,
   `Changed`, `Deprecated`, `Removed`, `Fixed`) instead of duplicating the
   release-note overview.
6. Within each category, group related entries by topic when simple reordering
   improves readability. Prefer clusters such as target layout, SDF/BVH/raycast,
   USD/importer, solver reset, viewer/rendering, dependency bumps, and examples
   over chronological or random ordering.
7. Audit category boundaries before finalizing. Keep `Added` for new public
   APIs, options, features, examples, and docs; move existing-API behavior
   changes, new warnings, default changes, and importer/solver semantics into
   `Changed`, even when they expand support.
8. Add same-repository PR references as compact `(#NNNN)` references
   selectively, not mechanically. Prioritize high-importance entries:
   breaking/default-changing behavior, public API additions that affect
   migration, deprecations, removals, and major support fixes. Do not add PR
   refs to every routine docs, example, cleanup, or minor fix entry.
9. Before adding a PR reference, verify that the PR actually introduced the
   change being cited. Prefer local history such as `git log --oneline` and
   `git show --name-only <commit>`; skip ambiguous references rather than
   guessing.
10. For each breaking, removed, deprecated, or default-changing entry, include
   migration guidance or a clear action: replacement symbol, opt-out flag,
   compatibility setting, or what to re-test.
11. Avoid directing users to private/internal APIs as migration targets. If a
   public alias is deprecated because storage is becoming internal, say to avoid
   depending on that data directly rather than pointing at underscore-prefixed
   members.
12. Separate internal cleanup from public API removals. If an internal symbol is
   mentioned for completeness, label it as internal and do not imply users must
   migrate unless it was public.
13. Verify restored APIs against the final/RC tag before classifying removals.
    For example, if a public symbol was removed during development but restored
    before the release tag, do not list it as removed.
14. When moving entries between release sections, make sure the information is
    not duplicated under an older released version and the historical section
    still reflects what actually shipped there.

## Checks

Run targeted searches before finishing:

```bash
rg -n "removed|removal|deprecated|will be removed|private|_[a-zA-Z].*in favor|SensorRaycast|raycast_kernel_no_hfield" CHANGELOG.md
git diff -- CHANGELOG.md
```

Review the diff for accidental deletion, duplicate entries across release
sections, stale fixed-version removal targets, and upgrade-impact entries that
lack migration guidance or a PR reference.
