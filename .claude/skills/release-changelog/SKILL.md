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

1. Protect released history before doing anything else. Resolve the latest
   stable tag, diff its `CHANGELOG.md` against the working ref, and inspect
   every hunk beneath any dated `## [X.Y.Z]` header. Treat any later addition,
   removal, rewording, or move inside a released section as misfiled until
   proven intentional. This commonly happens when a PR written before the
   release lands afterward and inserts its entry beneath the former
   `[Unreleased]` categories, which now belong to `X.Y.Z`. Move such entries to
   the current `[Unreleased]` section before continuing. Only change released
   history for a deliberate correction with explicit maintainer approval.
2. Identify the release ref and comparison base. For final releases, use the
   final tag or release branch. For RC prep, use the latest RC tag as temporary
   ground truth and verify against the previous released tag.
3. Read the current `CHANGELOG.md` section being edited, the release audit if
   one exists, and PRs behind unclear entries. Do not rely only on commit
   subjects for migration guidance.
4. Check completeness against every merged change from the previous GA or
   micro release through the release ref, including fixes merged during RC
   stabilization. Compare the commit and PR range with both the release audit
   and the current changelog; add user-visible changes that were missed.
5. Preserve information. Rephrase, split, merge, and regroup entries only when
   the facts remain intact. Ask before deleting information, omitting a
   questionable entry, or downgrading a user-visible change to silence.
6. Keep the changelog detail-oriented and use the existing canonical
   Keep-a-Changelog categories (`Added`, `Changed`, `Deprecated`, `Removed`,
   `Fixed`). Do not add a separate upgrade-notes or release-summary block;
   release notes and announcements carry the summary, while migration and
   retesting guidance belongs in the affected changelog entries.
7. Within each category, group related entries by user-facing feature area or
   migration theme when simple reordering improves readability. Derive those
   groups from the current release instead of carrying subsystem examples from
   an older release forward.
8. Remove exact and semantic duplicates within the release, not only identical
   wording. If a feature and a fix for that feature both landed during the same
   release cycle, consolidate the entries around the final user-visible
   behavior instead of recording it once as `Added` and again as `Fixed`.
9. Audit category boundaries before finalizing. Keep `Added` for new public
   APIs, options, features, examples, and docs; move existing-API behavior
   changes, new warnings, default changes, and importer/solver semantics into
   `Changed`, even when they expand support.
10. Add same-repository PR references as compact `(#NNNN)` references
   selectively, not mechanically. Prioritize high-importance entries:
   breaking/default-changing behavior, public API additions that affect
   migration, deprecations, removals, and major support fixes. Do not add PR
   refs to every routine docs, example, cleanup, or minor fix entry.
11. Before adding a PR reference, verify that the PR actually introduced the
   change being cited. Prefer local history such as `git log --oneline` and
   `git show --name-only <commit>`; skip ambiguous references rather than
   guessing.
12. For each breaking, removed, deprecated, or default-changing entry, include
   migration guidance or a clear action: replacement symbol, opt-out flag,
   compatibility setting, or what to re-test.
13. Avoid directing users to private/internal APIs as migration targets. If a
   public alias is deprecated because storage is becoming internal, say to avoid
   depending on that data directly rather than pointing at underscore-prefixed
   members.
14. Separate internal cleanup from public API removals. If an internal symbol is
   mentioned for completeness, label it as internal and do not imply users must
   migrate unless it was public.
15. Verify restored APIs against the final/RC tag before classifying removals.
    For example, if a public symbol was removed during development but restored
    before the release tag, do not list it as removed.
16. When moving entries between release sections, make sure the information is
    not duplicated under an older released version and the historical section
    still reflects what actually shipped there.
17. Perform a second editorial pass after regrouping. Re-read the source entries
    and the final diff to catch user-relevant behavior, limitations, opt-in
    conditions, changed defaults, compatibility details, or migration actions
    lost during condensation.

## Post-release reconciliation

After a release branch finalizes its changelog, merge that release section back
to `main` through a dedicated feature branch and a changelog-only PR:

1. Fetch the canonical remote and create the feature branch from the latest
   `upstream/main`, not from the release branch.
2. Treat the final tag (or `upstream/release-X.Y` before the tag is available)
   as the source of truth for the complete `## [X.Y.Z] - YYYY-MM-DD` section.
3. Keep a fresh `## [Unreleased]` section as the first version header on
   `main`. Preserve every post-cut entry not shipped in the release under the
   correct category; do not replace the whole file with the release-branch
   copy.
4. Insert the finalized release section immediately below `[Unreleased]` and
   keep shipped entries only in that dated section. Resolve semantic overlap so
   the same user-facing change is not recorded twice.
5. Verify the PR changes only `CHANGELOG.md`, the dated section matches the
   final tag, and older released sections remain unchanged.

If another maintainer is already preparing the merge-back, do not create a
competing changelog edit. Confirm the current `upstream/main` and coordinate on
the existing branch or PR instead.

## Checks

Run targeted searches before finishing:

```bash
git diff v<latest-release> -- CHANGELOG.md
git log --oneline <previous-release>..<release-ref>
rg -n "removed|removal|deprecated|will be removed|in favor of|use .* instead|renam|replac|default|breaking" CHANGELOG.md
git diff -- CHANGELOG.md
```

Review every changelog hunk since the latest release tag and confirm none lands
inside that or any older released section. Then review the release range for
missing entries and the working diff for accidental deletion, semantic
duplicates within or across release sections, stale fixed-version removal
targets, and upgrade-impact entries that lack migration guidance or a PR
reference.
