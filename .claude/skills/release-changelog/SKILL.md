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

1. Protect released history first. Diff `CHANGELOG.md` from the latest stable
   tag and inspect every hunk under a dated version header. Move late PR entries
   accidentally added to a released section into the current `[Unreleased]`
   section. Change released history only with explicit maintainer approval.
2. Identify the release ref and comparison base. For final releases, use the
   final tag or release branch. For RC prep, use the latest RC tag as temporary
   ground truth and verify against the previous released tag.
3. Read the current `CHANGELOG.md` section being edited, the release audit if
   one exists, and PRs behind unclear entries. Do not rely only on commit
   subjects for migration guidance.
4. Check completeness from the previous GA or micro release through the release
   ref, including RC fixes. Compare the range with the release audit and add
   missed user-visible changes.
5. Preserve information. Rephrase, split, merge, and regroup entries only when
   the facts remain intact. Ask before deleting information, omitting a
   questionable entry, or downgrading a user-visible change to silence.
6. Use the existing Keep-a-Changelog categories (`Added`, `Changed`,
   `Deprecated`, `Removed`, `Fixed`). Keep migration and retesting guidance in
   the affected entries; release notes carry the summary.
7. Within each category, group entries by the current release's user-facing
   feature areas or migration themes when this improves readability.
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

Merge a release branch's finalized changelog back to `main` through a dedicated
feature branch and changelog-only PR:

1. Fetch the canonical remote and create the feature branch from the latest
   `upstream/main`, not from the release branch.
2. Use the final tag as the source of truth.
3. Keep `## [Unreleased]` first and preserve all post-cut entries not shipped
   in the release. Do not replace the whole file with the release-branch copy.
4. Insert the finalized release section immediately below `[Unreleased]` and
   keep shipped entries only in that dated section. Resolve semantic overlap so
   the same user-facing change is not recorded twice.
5. Verify that only `CHANGELOG.md` changes, the dated section matches the final
   tag, and older released sections remain unchanged.

## Checks

Run targeted searches before finishing:

```bash
git diff v<latest-release> -- CHANGELOG.md
git log --oneline <previous-release>..<release-ref>
rg -n "removed|removal|deprecated|will be removed|in favor of|use .* instead|renam|replac|default|breaking" CHANGELOG.md
git diff -- CHANGELOG.md
```

Confirm that no new hunk lands in a released section, then check for missing or
duplicate entries, accidental deletions, stale removal targets, and missing
migration guidance or PR references.
