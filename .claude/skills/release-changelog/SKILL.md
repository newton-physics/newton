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
3. Check completeness against every merged change from the previous GA or
   micro release through the release ref, including fixes merged during RC
   stabilization. Compare the commit and PR range with both the release audit
   and the current changelog; add user-visible changes that were missed.
4. Preserve information. Rephrase, split, merge, and regroup entries only when
   the facts remain intact. Ask before deleting information, omitting a
   questionable entry, or downgrading a user-visible change to silence.
5. Keep the changelog detail-oriented and use the existing canonical
   Keep-a-Changelog categories (`Added`, `Changed`, `Deprecated`, `Removed`,
   `Fixed`). Do not add a separate upgrade-notes or release-summary block;
   release notes and announcements carry the summary, while migration and
   retesting guidance belongs in the affected changelog entries.
6. Within each category, group related entries by topic when simple reordering
   improves readability. Prefer clusters such as target layout, SDF/BVH/raycast,
   USD/importer, solver reset, viewer/rendering, dependency bumps, and examples
   over chronological or random ordering.
7. Remove exact and semantic duplicates within the release, not only identical
   wording. If a feature and a fix for that feature both landed during the same
   release cycle, consolidate the entries around the final user-visible
   behavior instead of recording it once as `Added` and again as `Fixed`.
8. Audit category boundaries before finalizing. Keep `Added` for new public
   APIs, options, features, examples, and docs; move existing-API behavior
   changes, new warnings, default changes, and importer/solver semantics into
   `Changed`, even when they expand support.
9. Add same-repository PR references as compact `(#NNNN)` references
   selectively, not mechanically. Prioritize high-importance entries:
   breaking/default-changing behavior, public API additions that affect
   migration, deprecations, removals, and major support fixes. Do not add PR
   refs to every routine docs, example, cleanup, or minor fix entry.
10. Before adding a PR reference, verify that the PR actually introduced the
   change being cited. Prefer local history such as `git log --oneline` and
   `git show --name-only <commit>`; skip ambiguous references rather than
   guessing.
11. For each breaking, removed, deprecated, or default-changing entry, include
   migration guidance or a clear action: replacement symbol, opt-out flag,
   compatibility setting, or what to re-test.
12. Avoid directing users to private/internal APIs as migration targets. If a
   public alias is deprecated because storage is becoming internal, say to avoid
   depending on that data directly rather than pointing at underscore-prefixed
   members.
13. Separate internal cleanup from public API removals. If an internal symbol is
   mentioned for completeness, label it as internal and do not imply users must
   migrate unless it was public.
14. Verify restored APIs against the final/RC tag before classifying removals.
    For example, if a public symbol was removed during development but restored
    before the release tag, do not list it as removed.
15. When moving entries between release sections, make sure the information is
    not duplicated under an older released version and the historical section
    still reflects what actually shipped there.
16. Perform a second editorial pass after regrouping. Re-read the source entries
    and the final diff to catch user-relevant behavior, limitations, opt-in
    conditions, changed defaults, compatibility details, or migration actions
    lost during condensation.

## Post-release reconciliation

When bringing a dated release section from the release branch back to `main`:

1. Start from the current `main` branch and import the dated release section;
   do not replace `main`'s changelog wholesale.
2. Keep a fresh `[Unreleased]` section at the top of `main`.
3. Preserve entries added to `main` after the release branch was cut but not
   shipped in the release. Move them into the new `[Unreleased]` section under
   the correct categories.
4. Keep shipped entries only in the dated release section. Resolve semantic
   overlap between that section and `[Unreleased]` so the same user-facing
   change is not recorded twice.
5. Verify that older released sections remain unchanged.

## Checks

Run targeted searches before finishing:

```bash
git log --oneline <previous-release>..<release-ref>
rg -n "removed|removal|deprecated|will be removed|private|_[a-zA-Z].*in favor|SensorRaycast|raycast_kernel_no_hfield" CHANGELOG.md
git diff -- CHANGELOG.md
```

Review the release range for missing entries. Review the diff for accidental
deletion, semantic duplicates within or across release sections, stale
fixed-version removal targets, and upgrade-impact entries that lack migration
guidance or a PR reference.
